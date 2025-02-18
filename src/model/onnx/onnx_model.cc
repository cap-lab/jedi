#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cctype>
#include <sstream>
#include <set>
#include <algorithm>
#include <thread>
#include <libconfig.h++>
#include <unordered_map>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "cuda_jedi.h"
#include "variable.h"
#include "util.h"


#include "onnx_model.h"

using namespace nvinfer1;

using namespace nvonnxparser;

REGISTER_JEDI_NETWORK_MODEL(OnnxModel);

#ifndef FatalError
#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}
#endif 

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        //if (severity <= Severity::kWARNING)
		std::cout <<"TENSORRT LOG: "<< msg << std::endl;
    }
} logger;

OnnxParserLogger onnx_logger;

static void loadFileToBuffer(std::string file_name, char* &buffer, size_t &size) {
	//char *gieModelStream{nullptr};
	//size_t size{0};
	buffer = nullptr;
	size = 0;
	std::ifstream file(file_name, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		buffer = new char[size];
		file.read(buffer, size);
		file.close();
	}
}

static void setUnnamedLayerAndTensorName(INetworkDefinition* &network, int start_cut_point) {
	int layer_num = network->getNbLayers();

	// change layer name first
	for(int index = 0 ; index < layer_num ; index++) {
		ILayer *layer = network->getLayer(index);
		std::string layerName = layer->getName();

		if(layerName.size() == 0 || layerName.rfind("(Unnamed Layer* ", 0) == 0 || layerName.rfind("ONNXTRT_", 0) == 0) {
			int global_layer_id = start_cut_point + index;
			std::string layer_name = "JEDI_" +  std::to_string(global_layer_id) + "_" + convertLayerTypeToString(layer);
			layer->setName(layer_name.c_str());

			int output_num = layer->getNbOutputs();
			for(int out_index = 0; out_index < output_num ; out_index++) {
				ITensor *tensor = layer->getOutput(out_index);
				if(tensor != nullptr) {
					std::string ori_tensor_name = tensor->getName();
					if((ori_tensor_name.size() == 0 || ori_tensor_name.compare("(Unnamed Layer* ") != 0) && (!tensor->isNetworkOutput()) )  {
						std::string tensor_name = std::string(layer->getName()) + "_output_" + std::to_string(out_index);
						tensor->setName(tensor_name.c_str());
						//std::cout << "tensor name: " <<  ori_tensor_name << " => " << tensor_name  << std::endl;
					}
				}
			}
			//std::cout << "layer name (" << index << ") : " << layerName << " => " << layer_name << std::endl;
		}
	}

	for(int index = 0 ; index < layer_num ; index++) {
		ILayer *layer = network->getLayer(index);

		int output_num = layer->getNbOutputs();
		for(int out_index = 0; out_index < output_num ; out_index++) {
			ITensor *tensor = layer->getOutput(out_index);
			std::string output_str = "_output_";
			if(tensor != nullptr) {
				std::string ori_tensor_name = tensor->getName();
				if((ori_tensor_name.size() == 0 || ori_tensor_name.rfind(output_str) == std::string::npos) && (!tensor->isNetworkOutput()) )  {
					std::string tensor_name = std::string(layer->getName()) + "_output_" + std::to_string(out_index);
					tensor->setName(tensor_name.c_str());
					//std::cout << "tensor name (layer: " << layer->getName() << "): " <<  ori_tensor_name << " => " << tensor_name  << std::endl;
				} else {
					//std::cout << "unchanged tensor name (layer: " << layer->getName() << "): " << tensor->getName()  << std::endl;
				}
			}
		}

	}
}


static void readOptimizationConfigFile(libconfig::Config *cfg, std::string config_file_path) {
	try {
		cfg->readFile(config_file_path.c_str());
	}
	catch(const libconfig::FileIOException &fioex) {
		std::cerr << "I/O error while reading file: " << config_file_path << std::endl;
		exit(0);
	}
	catch(const libconfig::ParseException &pex) {
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()<< " - " << pex.getError() << " of file " << config_file_path  <<  std::endl;
		exit(-1);
	}
}

static void setOptimizationDataFromCfg(IOptimizationProfile *profile, ITensor *tensor, libconfig::Setting &current_setting, std::string opt_string, OptProfileSelector selector) {
	Dims tensor_dim;
	const char *data = current_setting[opt_string.c_str()];
	std::stringstream ss(data);
	std::string temp;
	int dim_idx = 0;

	while(std::getline(ss,temp,',')) {
		tensor_dim.d[dim_idx] = std::stoi(temp);
		dim_idx++;
	}
	tensor_dim.nbDims = dim_idx;
	profile->setDimensions(tensor->getName(), selector, tensor_dim);
	std::cout << tensor->getName()  << " " << opt_string << ": " << tensor_dim.d[0] << ", " << tensor_dim.d[1]  << std::endl; 
}

static bool valueInRange(std::vector<LayerRange> ranges, int value) {
	bool valueInRange = false;
	for(unsigned int iter = 0; iter < ranges.size(); iter++) {
		if(value >= ranges[iter].start && value <= ranges[iter].end) {
			valueInRange = true;
			break;
		}
	}

	return valueInRange;
}

static std::string makeRangeString(int prev_cut_point, int curr_cut_point, std::vector<LayerRange> ranges) {
	unsigned int layerNum = curr_cut_point - prev_cut_point + 1;
	bool continueValue = false;
	bool first = true;
	std::string rangeString = "";

	for (unsigned int iter = 0 ; iter < layerNum  ; iter++) {
		bool valueIsIncluded = valueInRange(ranges, prev_cut_point + iter);
		if (valueIsIncluded == true) {
			if( continueValue == false) {
				if (first == false) {
					rangeString += ",";
				}
				rangeString += std::to_string(prev_cut_point + iter);
				first = true;
				continueValue = true;
			}
		} else {
			if (continueValue == true) {
				rangeString += "-" + std::to_string(prev_cut_point + iter);
				continueValue = false;
				first = false;
			}
		}
	}

	if(continueValue == true) {
		rangeString += "-" + std::to_string(prev_cut_point + layerNum - 1);
	}

	return rangeString;
}

static bool isQuantizedModel(int device, int data_type, bool hasQuantizedModel) {
	if (device == DEVICE_GPU && data_type == TYPE_INT8 && hasQuantizedModel == true) {
		return true;
	} else {
		return false;
	}
}

void OnnxModel::getModelFileName(int curr, std::string &plan_file_name, INetworkDefinition *network, std::string postfix, bool for_rt_build, bool is_quantized_model) {
	std::string model_dir = config_data->instances.at(instance_id).model_dir;
	std::string cut_points_name;
	std::string device_name;
	std::string data_type_name;
	std::string image_size_name;
	int device = config_data->instances.at(instance_id).devices.at(curr);
	int data_type = config_data->instances.at(instance_id).data_types.at(curr);
	int aux_stream_num = config_data->instances.at(instance_id).aux_stream_numbers.at(curr);
	int dla_sram_size = config_data->instances.at(instance_id).dla_sram_sizes.at(curr);
	std::string network_name = config_data->instances.at(instance_id).network_name;
	int prev_cut_point = 0, curr_cut_point = 0;
	std::vector<LayerRange> gpu_ranges = config_data->instances.at(instance_id).gpu_ranges;
	std::vector<LayerRange> fp16_ranges = config_data->instances.at(instance_id).fp16_ranges;
	std::vector<LayerRange> fp32_ranges = config_data->instances.at(instance_id).fp32_ranges;
	std::string rt_post_fix = "";
	
	if(curr > 0) {
		prev_cut_point = config_data->instances.at(instance_id).cut_points.at(curr-1) + 1;
	}
	curr_cut_point = config_data->instances.at(instance_id).cut_points.at(curr);

	cut_points_name = std::to_string(prev_cut_point) + "." + std::to_string(curr_cut_point);


	if (for_rt_build == true) {
		if(device == DEVICE_DLA) {
			device_name = "DLA";
		}
		else {
			device_name = "GPU";
		}

		if(data_type == TYPE_FP32) {
			data_type_name = "FP32";
		}
		else if(data_type == TYPE_FP16) {
			data_type_name = "FP16";
		}
		else if(data_type == TYPE_INT8) {
			data_type_name = "INT8";
		}
		rt_post_fix = "_" + device_name + "_" + data_type_name;
	}

	if (is_quantized_model == true && data_type == TYPE_INT8) {
		rt_post_fix += "_QUANT";
	}

	ITensor *tensor = network->getInput(0);
	Dims tensor_dim = tensor->getDimensions();
	
	std::string input_dim_name;
	for(int iter1 = 0 ; iter1 < tensor_dim.nbDims ; iter1++) {
		if(iter1 > 0)
			input_dim_name += "x";
		input_dim_name += std::to_string(tensor_dim.d[iter1]);
	}

	plan_file_name = model_dir + "/model_" + network_name + "_onnx_" + input_dim_name  + "_" + cut_points_name + rt_post_fix;
	if(for_rt_build == true) {
		std::string range_string;
		plan_file_name = plan_file_name + "_aux" + std::to_string(aux_stream_num);
		if (device == DEVICE_DLA) {
			plan_file_name = plan_file_name + "_sram" + std::to_string(dla_sram_size);

			range_string = makeRangeString(prev_cut_point, curr_cut_point, gpu_ranges);
			if(range_string.size() > 0) {
				plan_file_name += "_gpu" + range_string;
			}
		}

		if (is_quantized_model == false) {
			if (data_type == TYPE_INT8) {
				range_string = makeRangeString(prev_cut_point, curr_cut_point, fp16_ranges);
				if(range_string.size() > 0) {
					plan_file_name += "_half" + range_string;
				}
			}

			if (data_type == TYPE_INT8 || data_type == TYPE_FP16) {
				range_string = makeRangeString(prev_cut_point, curr_cut_point, fp32_ranges);
				if(range_string.size() > 0) {
					plan_file_name += "_float" + range_string;
				}
			}
		}
	}
	plan_file_name = plan_file_name + postfix;

	std::cerr<<"plan_file_name: "<< plan_file_name<<std::endl;
}


bool OnnxModel::saveLayerInfoFile(std::string filename, const char *layerInfo){
    std::ofstream p(filename);
    if (!p) {
		std::cerr << "Could not open file: " << filename  << std::endl;
		return false;
    }

    if(layerInfo == nullptr)
		std::cerr << "Can't write layer info to file: " << filename  << std::endl;

    p.write(layerInfo, strlen(layerInfo));
    return true;
}

bool OnnxModel::serialize(const char *filename, nvinfer1::IHostMemory *ptr){
    std::ofstream p(filename, std::ios::binary);
    if (!p) {
		std::cerr << "Could not open file: " << filename  << std::endl;
		return false;
    }

    if(ptr == nullptr)
		std::cerr << "Can't serialize data file: " << filename  << std::endl;

    p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
    return true;
}

void OnnxModel::getIOTensorNamesOfLayer(INetworkDefinition *network, int layer_id, std::vector<std::string>& tensor_name_vec, bool is_input) {
	ILayer *layer = network->getLayer(layer_id);

	if(is_input) {
		for (int iter1 = 0; iter1 < layer->getNbInputs(); iter1++) {
			ITensor *tensor = layer->getInput(iter1);	
			tensor_name_vec.push_back(tensor->getName());
		}
	}
	else {
		for (int iter1 = 0; iter1 < layer->getNbOutputs(); iter1++) {
			ITensor *tensor = layer->getOutput(iter1);	
			tensor_name_vec.push_back(tensor->getName());
		}
	}
}

bool OnnxModel::checkTensorIsUsedInNextStages(int device_id, INetworkDefinition *network, int end_index, int layer_id, std::string tensor_name) {
	int device_num = config_data->instances.at(instance_id).device_num;
	int layer_num = network->getNbLayers();

	if(device_id+1 == device_num) {
		ILayer *layer = network->getLayer(layer_id);
		for (int iter2 = 0; iter2 < layer->getNbOutputs(); iter2++) {
			ITensor *tensor = layer->getOutput(iter2);
			if(tensor->isNetworkOutput()) {
				return true;	
			}
		}
	}

	for (int iter1 = end_index+1; iter1 < layer_num; iter1++) {
		ILayer *layer = network->getLayer(iter1);

		for (int iter2 = 0; iter2 < layer->getNbInputs(); iter2++) {
			ITensor *tensor = layer->getInput(iter2);
			if(tensor != nullptr) {
				//std::cerr<<"tensor name: "<<tensor->getName()<<", layer_id: "<<iter1<<", target tensor name: "<<tensor_name<<std::endl;
				//std::cerr<<"tensor isNetworkOutput: "<<tensor->isNetworkOutput()<<std::endl;
				if(tensor_name.compare(tensor->getName()) == 0) {
					return true;					
				}
			}
		}
	}

	return false;
}

void OnnxModel::getOutputIndexOfStage(int device_id, INetworkDefinition *network, int start_index, int end_index, std::vector<int>& output_index_vec, int dequantize_skip_index) {
	for (int iter1 = start_index ; iter1 <= end_index; iter1++) {
		ILayer *layer = network->getLayer(iter1);
		bool inserted = false;
		if (layer->getType() != nvinfer1::LayerType::kCONSTANT && (layer->getType() != nvinfer1::LayerType::kDEQUANTIZE || iter1 >= dequantize_skip_index)) {
			for (int iter2 = 0; iter2 < layer->getNbOutputs(); iter2++) {
				ITensor *tensor = layer->getOutput(iter2);
				std::string tensor_name = tensor->getName();

				if (inserted == false && checkTensorIsUsedInNextStages(device_id, network, end_index, iter1, tensor_name)) {
					output_index_vec.push_back(iter1);
					inserted = true;
				}
			}
		}
	}
}

void OnnxModel::fillInputs(int device_id, INetworkDefinition *network, int start_index, int end_index, std::vector<std::string>& input_name_vec, int dequantize_skip_index) {
	std::set<ITensor *> output_set;
	for(int iter1 = 0 ; iter1 < start_index ; iter1++) {
		ILayer *layer = network->getLayer(iter1);
		if (layer->getType() != nvinfer1::LayerType::kCONSTANT && (layer->getType() != nvinfer1::LayerType::kDEQUANTIZE || iter1 >= dequantize_skip_index)) {
			for (int iter2 = 0; iter2 < layer->getNbOutputs(); iter2++) {
				ITensor *tensor = layer->getOutput(iter2);
				if (output_set.find(tensor) == output_set.end()) {
					output_set.insert(tensor);
				}
			}
		}
	}

	for(int iter1 = start_index ; iter1 <= end_index ; iter1++) {
		ILayer *layer = network->getLayer(iter1);
		for(int iter2 = 0 ; iter2 < layer->getNbInputs() ; iter2++) {
			ITensor *tensor = layer->getInput(iter2);
			if(output_set.find(tensor) != output_set.end()) {
				std::cout << "add tensor1: " << tensor->getName() << std::endl;
				input_name_vec.push_back(tensor->getName());
				output_set.erase(tensor);
			}
			else if(tensor != NULL && tensor->isNetworkInput() == true) {
				auto it = std::find(input_name_vec.begin(), input_name_vec.end(), tensor->getName());
				if (it == input_name_vec.end()) {
					std::cout << "add tensor2: " << tensor->getName() << std::endl;

					input_name_vec.push_back(tensor->getName());
				}
			}
			//std::cout << "merong: " << iter1 << ", " << iter2 << std::endl;
		}
	}

}

void OnnxModel::surgeonOnnxByPolygraphy(int device_id, INetworkDefinition *network, std::string model_name, std::string onnx_file_name, int start_index, int end_index, int dequantize_skip_index) {
	int result = -1;
	std::string cmd = "polygraphy surgeon extract " + model_name + " -o " + onnx_file_name;
	std::string inputs = " --inputs ";
	std::string outputs = " --outputs ";

	std::vector<std::string> input_name_vec, output_name_vec;
	std::vector<int> output_index_vec;


	//getIOTensorNamesOfLayer(network, start_index, input_name_vec, true);
	fillInputs(device_id, network, start_index, end_index, input_name_vec, dequantize_skip_index);
	getOutputIndexOfStage(device_id, network, start_index, end_index, output_index_vec, dequantize_skip_index);
	std::cerr<<"output_index_vec size: "<<output_index_vec.size()<<std::endl;
	for(int output_index : output_index_vec) {
		std::cerr<<"\toutput_index: "<<output_index<<std::endl;
		getIOTensorNamesOfLayer(network, output_index, output_name_vec, false);
	}

	for(auto name :input_name_vec) {
		inputs.append("\"" + name + ":auto:auto\" ");	
	}
	for(auto name :output_name_vec) {
		outputs.append("\"" + name + ":auto\" ");	
	}

	cmd = cmd + inputs + outputs;
	std::cerr<<"cmd: "<<cmd<<std::endl;
	result = system(cmd.c_str());
	if(result == -1 || result == 127) {
		std::cerr<<"ERROR occurs at "<<__func__<<":"<<__LINE__<<std::endl;	
	}
}

static int convertCutpointIndexFromOriginalToQuantizedModel(INetworkDefinition *network, INetworkDefinition *quantized_network, int cut_point) {
	ILayer *original_layer = network->getLayer(cut_point);
	int cut_index = -1;
	std::string layer_name = original_layer->getName();
	int quantized_layer_num = 0;

	quantized_layer_num = quantized_network->getNbLayers();
	// assume that the quantized layer num is greater than original layer num
	for (int index = 0; index < quantized_layer_num; index++) {
		cut_index = (cut_point + index) % quantized_layer_num;
		ILayer *layer = quantized_network->getLayer(cut_index);
		//printf("layer name: %s\n", layer->getName());
		if (layer_name.compare(layer->getName()) == 0) {
			printf("layer name: %s\n", layer->getName());
			break;
		}
	}

	return cut_index;
}

void OnnxModel::separateOnnxFile(INetworkDefinition *network, std::string model_name, std::string quantized_model_name, std::vector<std::string>& onnx_file_name_vec) {
	int device_num = config_data->instances.at(instance_id).device_num;
	int prev_cut_point = 0, curr_cut_point = 0;
	INetworkDefinition *quantized_network = nullptr;
	IBuilder *quant_builder = nullptr;
	int error_num = 0;
	int dequantize_skip_index = 0;

	if (quantized_model_name.length() > 0) {
		quant_builder = createInferBuilder(onnx_logger);
		quantized_network =  quant_builder->createNetworkV2(0);
		IParser* parser = createParser(*quantized_network, onnx_logger);
		parser->parseFromFile(quantized_model_name.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
		error_num = parser->getNbErrors();
		for (int32_t i = 0; i < error_num; ++i)
		{
			std::cout << "TENSORRT ONNX ERROR: "  << parser->getError(i)->desc() << std::endl;
		}

		if(error_num > 0) {
			FatalError("Onnx parsing failed");
		}
		dequantize_skip_index = convertCutpointIndexFromOriginalToQuantizedModel(network, quantized_network, 0);
	}
	else {
		dequantize_skip_index = convertCutpointIndexFromOriginalToQuantizedModel(network, network, 0);
	}

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		std::string onnx_file_name;
		int data_type = config_data->instances.at(instance_id).data_types.at(iter1);
		int device = config_data->instances.at(instance_id).devices.at(iter1);

		getModelFileName(iter1, onnx_file_name, network, ".onnx", false, isQuantizedModel(device, data_type, quantized_model_name.length() > 0));

		onnx_file_name_vec.push_back(onnx_file_name);
		if(iter1 > 0) {
			prev_cut_point = curr_cut_point + 1;
		}

		curr_cut_point = config_data->instances.at(instance_id).cut_points.at(iter1);
		curr_cut_point = std::min(curr_cut_point, network->getNbLayers()-1);

		if(fileExist(onnx_file_name) == false)  {
			if (quantized_model_name.length() > 0 && data_type == TYPE_INT8 && device == DEVICE_GPU) {
				int prev_cut_point_changed = 0, curr_cut_point_changed = 0;
				if (prev_cut_point > 0) {
					// Since the +1 value of the cutpoint from the original onnx file and the +1 value of the cutpoint from quantized onnx file can be different, 
					// we convert the last cutpoint first, and then add 1 to the converted cutpoint.
					// minus 1 to become the last cutpoint of previous stage
					prev_cut_point--;
					prev_cut_point_changed = convertCutpointIndexFromOriginalToQuantizedModel(network, quantized_network, prev_cut_point);
					// add 1 to become the first cutpoint of the current stage
					prev_cut_point_changed++;
				}
				if (curr_cut_point > 0) {
					curr_cut_point_changed = convertCutpointIndexFromOriginalToQuantizedModel(network, quantized_network, curr_cut_point);
				}
				surgeonOnnxByPolygraphy(iter1, quantized_network, quantized_model_name, onnx_file_name, prev_cut_point_changed, curr_cut_point_changed, dequantize_skip_index);
			} else {
				surgeonOnnxByPolygraphy(iter1, network, model_name, onnx_file_name, prev_cut_point, curr_cut_point, dequantize_skip_index);
			}
		}
	}
}

static void updateLayerAndOutputType(ILayer *layer, nvinfer1::DataType updatedType) {
	int old_precision = (int) layer->getPrecision();
	layer->setPrecision(updatedType);
	std::cout << "Precision changed " << layer->getName() << ": " << old_precision << " => " << (int) layer->getPrecision() << std::endl;
	int output_num = layer->getNbOutputs();
	for(int output_index = 0 ; output_index < output_num  ; output_index++) {
		nvinfer1::DataType output_type = layer->getOutputType(output_index);
#if NV_TENSORRT_MAJOR > 8
		if (output_type != nvinfer1::DataType::kINT64) {
#else
		if (output_type != nvinfer1::DataType::kINT32) {
#endif
			layer->setOutputType(output_index, updatedType);
			//layer->getOutput(output_index)->setType(updatedType);
		}
	}
}

static void addNetworkOutputNodeToLeakyRelu(ILayer *layer, INetworkDefinition *network, std::unordered_map<ITensor *, ITensor *> &tensorsTobeChanged) {
	int output_num = layer->getNbOutputs();
	for(int output_index = 0 ; output_index < output_num  ; output_index++) {
		if(layer->getOutput(output_index)->isNetworkOutput()) {
			ITensor *tensor = layer->getOutput(output_index);
			IActivationLayer *iActLayer = network->addActivation(*tensor, nvinfer1::ActivationType::kLEAKY_RELU);
			iActLayer->setAlpha(1.0f);

			std::string old_output = tensor->getName();
			std::string new_output = tensor->getName();
			new_output += "_old";
			tensor->setName(new_output.c_str());
			//iActLayer->setPrecision(nvinfer1::DataType::kHALF);
			iActLayer->getOutput(0)->setName(old_output.c_str());
			network->markOutput(*(iActLayer->getOutput(0)));
			network->unmarkOutput(*(layer->getOutput(output_index)));
			tensorsTobeChanged[tensor] = iActLayer->getOutput(0);
			std::cout << "mark output old: " << new_output << std::endl;
		}
	}
}


IBuilderConfig* OnnxModel::createEngineFromOnnxFile(int cur_iter, std::string onnx_file_name, bool is_quantized_onnx, IBuilder* &builder, INetworkDefinition* &network, IParser* &parser) {
	int data_type = config_data->instances.at(instance_id).data_types.at(cur_iter);
	int device = config_data->instances.at(instance_id).devices.at(cur_iter);
	std::vector<LayerRange> gpu_ranges = config_data->instances.at(instance_id).gpu_ranges;
	std::vector<LayerRange> fp16_ranges = config_data->instances.at(instance_id).fp16_ranges;
	std::vector<LayerRange> fp32_ranges = config_data->instances.at(instance_id).fp32_ranges;

	int start_cut_point = 0;

	if(cur_iter > 0 )
		start_cut_point = config_data->instances.at(instance_id).cut_points.at(cur_iter - 1) + 1;

	builder = createInferBuilder(logger);

#if NV_TENSORRT_MAJOR > 8
	uint32_t flag = 0;
#else
	uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); // deprecated in tensorrt 10
#endif

	network =  (builder)->createNetworkV2(flag);

	parser = createParser(*network, logger);

	(parser)->parseFromFile(onnx_file_name.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < (parser)->getNbErrors(); ++i)
	{
		std::cout << "TENSORRT ONNX ERROR: "  << parser->getError(i)->desc() << std::endl;
	}

	if((parser)->getNbErrors() > 0) {
		FatalError("Onnx parsing failed");
	}

	setUnnamedLayerAndTensorName(network, start_cut_point);

	IBuilderConfig* config = builder->createBuilderConfig();

	int layer_num = network->getNbLayers();

	for(int index = 0 ; index < layer_num ; index++) {
		ILayer *layer = network->getLayer(index);
		if(device == DEVICE_DLA && valueInRange(gpu_ranges, start_cut_point + index) == true) {
			config->setDeviceType(layer, nvinfer1::DeviceType::kGPU);
		}
		if (is_quantized_onnx == false) {
			if(data_type == TYPE_INT8 && valueInRange(fp16_ranges, start_cut_point + index) == true) {
#if NV_TENSORRT_MAJOR > 8
				if(layer->getOutputType(0) != nvinfer1::DataType::kINT64 && layer->getType() != LayerType::kPLUGIN &&
				layer->getType() != LayerType::kPLUGIN_V2 && layer->getType() != LayerType::kPLUGIN_V3 /* && layer->getType() != LayerType::kSHUFFLE*/) {
#else
				if(layer->getOutputType(0) != nvinfer1::DataType::kINT32 && layer->getType() != LayerType::kPLUGIN &&
				layer->getType() != LayerType::kPLUGIN_V2) {
#endif
					updateLayerAndOutputType(layer, nvinfer1::DataType::kHALF);
				}
			}
			if((data_type == TYPE_INT8 || data_type == TYPE_FP16) && valueInRange(fp32_ranges, start_cut_point + index) == true) {
#if NV_TENSORRT_MAJOR > 8
				if(layer->getOutputType(0) != nvinfer1::DataType::kINT64 && layer->getType() != LayerType::kPLUGIN &&
				layer->getType() != LayerType::kPLUGIN_V2 && layer->getType() != LayerType::kPLUGIN_V3 /* && layer->getType() != LayerType::kSHUFFLE*/) {
#else
				if(layer->getOutputType(0) != nvinfer1::DataType::kINT32 && layer->getType() != LayerType::kPLUGIN &&
				layer->getType() != LayerType::kPLUGIN_V2) {
#endif
					updateLayerAndOutputType(layer, nvinfer1::DataType::kFLOAT);
				}
			}
		}
	}

	for(int index = 0 ; index < layer_num ; index++) {
		ILayer *layer = network->getLayer(index);
		std::string layerName = layer->getName();
		if(layer->getType() != LayerType::kCONSTANT) {
			if(data_type == TYPE_INT8) {
				if(layer->getType() == nvinfer1::LayerType::kCONVOLUTION){
					//layer->setPrecision( nvinfer1::DataType::kHALF);
				}

				if(/*layer->getType() == LayerType::kSLICE ||*/ layer->getType() == LayerType::kMATRIX_MULTIPLY/* || layer->getType() == LayerType::kSHUFFLE*/) {
					//layer->setPrecision( nvinfer1::DataType::kFLOAT);
					layer->setPrecision( nvinfer1::DataType::kHALF);
				}
				else if(layer->getType() == LayerType::kACTIVATION && index > 0 && network->getLayer(index-1)->getType() == LayerType::kSLICE) {
					//layer->setPrecision( nvinfer1::DataType::kHALF);
				}
				else {
					//layer->setPrecision( nvinfer1::DataType::kINT8);
				}
			}
		}
	}


	if(data_type == TYPE_INT8 && device == DEVICE_GPU) {
		for(int index = 0 ; index < layer_num ; index++) {
			ILayer *layer = network->getLayer(index);
			if(layer->getType() == nvinfer1::LayerType::kPOOLING) {
				IPoolingLayer *poolLayer = (IPoolingLayer *) layer;
				if(poolLayer->getPoolingType() == PoolingType::kMAX){
					layer->setPrecision( nvinfer1::DataType::kHALF);
				}
			}

			/*int output_num = layer->getNbOutputs();
			for(int out_index = 0; out_index < output_num ; out_index++) {
				ITensor *tensor = layer->getOutput(out_index);
				if(tensor != nullptr && tensor->isNetworkOutput()) {
					if(layer->getType() != nvinfer1::LayerType::kSHUFFLE) {
						layer->setPrecision(nvinfer1::DataType::kHALF);
					}
					//else {
					//	layer->setPrecision(nvinfer1::DataType::kINT32);
					//}
					std::cout << "precision printing: " << (int) layer->getPrecision() << std::endl;
					break;
				}
			}*/
		}
	}


	if(data_type == TYPE_INT8 && device == DEVICE_DLA) {
		for(int index = 0 ; index < layer_num ; index++) {
			ILayer *layer = network->getLayer(index);
			if(layer->getType() == nvinfer1::LayerType::kCONVOLUTION){
				IConvolutionLayer *convLayer = (IConvolutionLayer *) layer;
				Dims pad_dim = convLayer->getPaddingNd();
				for (int pad_index = 0 ; pad_index < pad_dim.nbDims ; pad_index++) {
					// Since DLA INT8 with convolution layer with padding (3,3) drops the accuracy of the network,
					// We forcely changes this convolution layer to FP16 to prevent accuracy drop of this issue.
					if(pad_dim.d[pad_index] >= 3) {
						layer->setPrecision(nvinfer1::DataType::kHALF);
						break;
					}
				}
			}
		}
	}

	if (data_type == TYPE_INT8 && device == DEVICE_GPU) {
		int net_output_num = network->getNbOutputs();
		std::unordered_map<ITensor *, ITensor *> tensorsTobeChanged;

		for (int index = 0 ; index < net_output_num ; index++) {
			printf("output(%d): %s\n", index, network->getOutput(index)->getName());
		}

		for(int index = 0 ; index < layer_num ; index++) {
			ILayer *layer = network->getLayer(index);
			if(layer->getType() == nvinfer1::LayerType::kELEMENTWISE && index > 0){
				ILayer *prevlayer = network->getLayer(index-1);
				if(prevlayer->getType() == nvinfer1::LayerType::kACTIVATION || layer->getType() == nvinfer1::LayerType::kELEMENTWISE) {
					addNetworkOutputNodeToLeakyRelu(layer, network, tensorsTobeChanged);
				}
			}
		}

		for(int index = 0 ; index < layer_num ; index++) {
			ILayer *layer = network->getLayer(index);
			int input_num = layer->getNbInputs();
			for(int input_index = 0 ; input_index < input_num  ; input_index++) {
				ITensor *tensor = layer->getInput(input_index);
				if(tensorsTobeChanged.find(tensor) != tensorsTobeChanged.end()) {
					layer->setInput(input_index, *(tensorsTobeChanged.find(tensor)->second));
					break;
				}
			}
		}

		for (int index = 0 ; index < net_output_num ; index++) {
			printf("output(%d): %s\n", index, network->getOutput(index)->getName());
		}
	}

	/*if ((data_type == TYPE_FP16 || data_type == TYPE_INT8) &&  cur_iter > 0) {
		int prev_data_type = config_data->instances.at(instance_id).data_types.at(cur_iter - 1);
		if(prev_data_type == TYPE_FP16 || prev_data_type == TYPE_INT8) {
			int input_num = network->getNbInputs();
			for (int  index = 0 ; index < input_num ; index++) {
				ITensor *tensor = network->getInput(index);
				//tensor->setType(nvinfer1::DataType::kHALF);
				tensor->setType(nvinfer1::DataType::kFLOAT);

			}
		}
	}*/

	/*if ((data_type == TYPE_FP16 || data_type == TYPE_INT8) &&  device_num > 1 && cur_iter < device_num - 1) {
		int next_data_type = config_data->instances.at(instance_id).data_types.at(cur_iter + 1);
		if (next_data_type == TYPE_FP16 || data_type == TYPE_INT8) {
			int output_num = network->getNbOutputs();
			for (int  index = 0 ; index < output_num ; index++) {
				ITensor *tensor = network->getOutput(index);
				//tensor->setType(nvinfer1::DataType::kHALF);
				tensor->setType(nvinfer1::DataType::kFLOAT);
				std::cerr << "output type set to FP16: " << tensor->getName() << std::endl;
			}
		}
	}*/

	return config;
}

void OnnxModel::loadTimingCache(IBuilderConfig* config, ITimingCache* &cache) {
	std::string timing_cache_path = config_data->instances.at(instance_id).timing_cache_path;

	if(fileExist(timing_cache_path) == false)  {
		cache = config->createTimingCache(nullptr, 0);
	}
	else {
		char *buffer = nullptr;
		size_t size = 0;
		loadFileToBuffer(timing_cache_path, buffer, size);
		cache = config->createTimingCache(buffer, size);
	}
}

void OnnxModel::saveTimingCache(ITimingCache *cache) {
	std::string timing_cache_path = config_data->instances.at(instance_id).timing_cache_path;

	IHostMemory *serializedCache = cache->serialize();
	serialize(timing_cache_path.c_str(), serializedCache);
	delete serializedCache;

}

void OnnxModel::printModel() {
	TensorRTNetwork *tensorrt_network = nullptr;

	tensorrt_network = dynamic_cast<TensorRTNetwork *>(app->createNetwork(&(config_data->instances.at(instance_id))));
	setUnnamedLayerAndTensorName(tensorrt_network->network, 0);
	tensorrt_network->printNetwork();

	delete tensorrt_network->network;
	delete tensorrt_network->builder;

	if(tensorrt_network->calibrator != nullptr ) {
		delete tensorrt_network->calibrator;
	}
	delete tensorrt_network;
}


void OnnxModel::initializeModel() {
	int device_num = config_data->instances.at(instance_id).device_num;
	TensorRTNetwork *tensorrt_network = nullptr;
	IBuilder *builder = nullptr;
	INetworkDefinition *network = nullptr;

	int start_index = 0;

	tensorrt_network = dynamic_cast<TensorRTNetwork *>(app->createNetwork(&(config_data->instances.at(instance_id))));
	builder = tensorrt_network->builder;
	network = tensorrt_network->network;
	setUnnamedLayerAndTensorName(network, 0);
	tensorrt_network->printNetwork();

	std::vector<std::string> onnx_file_name_vec;	
	separateOnnxFile(network, tensorrt_network->onnx_file_path, tensorrt_network->quantized_onnx_file_path, onnx_file_name_vec);

	libconfig::Config cfg;
	libconfig::Setting* setting_ptr = nullptr;
	if(tensorrt_network->optimization_cfg_path.size() > 0) {
	    readOptimizationConfigFile(&cfg, tensorrt_network->optimization_cfg_path );
		setting_ptr = &cfg.lookup("configs");
	}

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		int cut_point = config_data->instances.at(instance_id).cut_points[iter1];
		int dla_core = config_data->instances.at(instance_id).dla_cores[iter1];
		int device = config_data->instances.at(instance_id).devices.at(iter1);
		int data_type = config_data->instances.at(instance_id).data_types.at(iter1);
		int aux_stream_num = config_data->instances.at(instance_id).aux_stream_numbers.at(iter1);
		int dla_sram_size = config_data->instances.at(instance_id).dla_sram_sizes.at(iter1);
		bool save_layer_info = config_data->instances.at(instance_id).save_layer_info;
		bool engineBuilt = false;
		bool is_quantized_onnx = isQuantizedModel(device, data_type, tensorrt_network->quantized_onnx_file_path.length() > 0);
		std::string plan_file_name;
		getModelFileName(iter1, plan_file_name, network, ".rt", true, is_quantized_onnx);

		if(fileExist(plan_file_name) == false)  {
			IBuilder *partial_builder;
			INetworkDefinition *partial_network;
			IParser *partial_parser;
			IBuilderConfig* config = createEngineFromOnnxFile(iter1, onnx_file_name_vec[iter1], is_quantized_onnx, partial_builder, partial_network, partial_parser);

			config->setAvgTimingIterations(8);
			config->setMaxAuxStreams(aux_stream_num);
			config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1UL << 32UL);
			config->setFlag(BuilderFlag::kDEBUG);
			ITimingCache *cache = nullptr;
			loadTimingCache(config, cache);
			config->setTimingCache(*cache, false);
			config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
			config->setFlag(BuilderFlag::kSPARSE_WEIGHTS);
			config->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
			//config->setFlag(BuilderFlag::kSTRICT_NANS);

			if(save_layer_info == true) {
				config->setProfilingVerbosity( nvinfer1::ProfilingVerbosity::kDETAILED);
			}

			IOptimizationProfile* profile = partial_builder->createOptimizationProfile();
			if(setting_ptr != nullptr) {
				for(int iter2 = 0; iter2 < partial_network->getNbInputs(); iter2++) {
					ITensor *tensor = partial_network->getInput(iter2);
					libconfig::Setting &setting = *setting_ptr;
					if (setting.exists(tensor->getName())) {
						libconfig::Setting &current_setting = setting[tensor->getName()];
						setOptimizationDataFromCfg(profile, tensor, current_setting, "min", OptProfileSelector::kMIN);
						setOptimizationDataFromCfg(profile, tensor, current_setting, "opt", OptProfileSelector::kOPT);
						setOptimizationDataFromCfg(profile, tensor, current_setting, "max", OptProfileSelector::kMAX);
					}
					else {
						Dims tensor_dim = tensor->getDimensions();

						std::cout << tensor->getName()  << " set to default: " << tensor_dim.d[0] << ", " << tensor_dim.d[1]  << std::endl; 

						profile->setDimensions(tensor->getName(), OptProfileSelector::kMIN, tensor_dim);
						profile->setDimensions(tensor->getName(), OptProfileSelector::kOPT, tensor_dim);
						profile->setDimensions(tensor->getName(), OptProfileSelector::kMAX, tensor_dim);
					}
				}
			}
			else {
				for(int iter2 = 0; iter2 < partial_network->getNbInputs(); iter2++) {
					ITensor *tensor = partial_network->getInput(iter2);
					Dims tensor_dim = tensor->getDimensions();
					profile->setDimensions(tensor->getName(), OptProfileSelector::kMIN, tensor_dim);
					profile->setDimensions(tensor->getName(), OptProfileSelector::kOPT, tensor_dim);
					profile->setDimensions(tensor->getName(), OptProfileSelector::kMAX, tensor_dim);
				}
			}
			//config->setBuilderOptimizationLevel(3);
			config->addOptimizationProfile(profile);

			// DLA options	
			if (device == DEVICE_DLA) {
				config->setFlag(BuilderFlag::kFP16);
				config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
				if(dla_core < 2) {
					config->setDLACore(dla_core);
				}
				else {
					config->setDLACore(0);

				}
				config->setFlag(BuilderFlag::kGPU_FALLBACK);
				// config->setFlag(BuilderFlag::kSTRICT_TYPES);
				config->setFlag(BuilderFlag::kDIRECT_IO);
				config->setFlag(BuilderFlag::kREJECT_EMPTY_ALGORITHMS);

				config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM, (1U << 10) * dla_sram_size);
			}

			if(data_type == TYPE_FP16 && partial_builder->platformHasFastFp16()) {
				config->setFlag(BuilderFlag::kFP16);
			}
			else if(data_type == TYPE_INT8 && partial_builder->platformHasFastInt8()) {  	// int8 option
				if(partial_builder->platformHasFastFp16()) {
					config->setFlag(BuilderFlag::kFP16);
				}
				config->setFlag(BuilderFlag::kINT8);
				config->setCalibrationProfile(profile);
				config->setInt8Calibrator(tensorrt_network->calibrator);
			}
			unsigned int n = std::thread::hardware_concurrency();
			partial_builder->setMaxThreads(std::max((unsigned int) 1, n/2));

			IHostMemory *serializedModel = partial_builder->buildSerializedNetwork(*partial_network, *config);
			assert(serializedModel != nullptr);

			serialize(plan_file_name.c_str(), serializedModel);
			engineBuilt = true;
			saveTimingCache(cache);

			delete partial_parser;
			delete partial_network;
			delete config;
			delete partial_builder;

			delete serializedModel;
			delete cache;
		}
		
		Stage *stage = new Stage(config_data, instance_id, iter1, start_index, cut_point);
		int duplication_num = dla_core <= 1 ? 1 : std::max(dla_core, DLA_NUM); 
		for(int iter2 = 0; iter2 < duplication_num; iter2++) {
			int core = dla_core <= 1 ? dla_core : iter2 % DLA_NUM;

			char *gieModelStream{nullptr};
			size_t size{0};
			std::ifstream file(plan_file_name, std::ios::binary);
			if (file.good()) {
				file.seekg(0, file.end);
				size = file.tellg();
				file.seekg(0, file.beg);
				gieModelStream = new char[size];
				file.read(gieModelStream, size);
				file.close();
			}

			IRuntime* runtime = createInferRuntime(logger);
			if(device == DEVICE_DLA) {
				runtime->setDLACore(core);
			}
			//runtime->setMaxThreads(4);
			ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream, size);
			assert(engine != nullptr);
			std::string layer_info_file_name = plan_file_name + ".trt.json";
			if(save_layer_info == true && engineBuilt == true) {
				auto inspector = std::unique_ptr<IEngineInspector>(engine->createEngineInspector());
				//std::cout << inspector->getLayerInformation(0, LayerInformationFormat::kJSON); // Print the information of the first layer in the engine.
				saveLayerInfoFile(layer_info_file_name, inspector->getEngineInformation(LayerInformationFormat::kJSON));
			}

			stage->engines.push_back(engine);

			if (gieModelStream) delete [] gieModelStream;

			runtimes.push_back(runtime);
		}
		stages.push_back(stage);

		start_index = cut_point + 1;

	}

	delete network;
	delete builder;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		Stage *stage = stages[iter1];
		stage->createExecutionContext();
	}

	if(tensorrt_network->calibrator != nullptr ) {
		delete tensorrt_network->calibrator;
	}

	delete tensorrt_network;
}

void OnnxModel::finalizeModel() {
	for(unsigned int iter1 = 0; iter1 < stages.size(); iter1++) {
		Stage *stage = stages[iter1];
		stage->finalizeStage();
	}

	for(unsigned int iter1 = 0; iter1 < runtimes.size(); iter1++) {
		delete runtimes[iter1];
	}
}


