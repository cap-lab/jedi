#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cctype>
#include <sstream>
#include <set>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "cuda.h"
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

void OnnxModel::getModelFileName(int curr, std::string &plan_file_name, INetworkDefinition *network, std::string postfix) {
	std::string model_dir = config_data->instances.at(instance_id).model_dir;
	std::string cut_points_name;
	std::string device_name;
	std::string data_type_name;
	std::string image_size_name;
	int device = config_data->instances.at(instance_id).devices.at(curr);
	int data_type = config_data->instances.at(instance_id).data_types.at(curr);
	int prev_cut_point = 0, curr_cut_point = 0;
	
	if(curr > 0) {
		prev_cut_point = config_data->instances.at(instance_id).cut_points.at(curr-1) + 1;
	}
	curr_cut_point = config_data->instances.at(instance_id).cut_points.at(curr);

	cut_points_name = std::to_string(prev_cut_point) + "." + std::to_string(curr_cut_point);

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

	ITensor *tensor = network->getInput(0);
	Dims tensor_dim = tensor->getDimensions();
	
	total_input_size = 1;
	std::string input_dim_name;
	for(int iter1 = 0 ; iter1 < tensor_dim.nbDims ; iter1++) {
		if(iter1 > 0)
			input_dim_name += "x";
		input_dim_name += std::to_string(tensor_dim.d[iter1]);
	}


	plan_file_name = model_dir + "/model_onnx_" + input_dim_name  + "_" + cut_points_name + "_" + device_name + "_" + data_type_name + postfix;
	std::cerr<<"plan_file_name: "<<plan_file_name<<std::endl;
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

void OnnxModel::getOutputIndexOfStage(int device_id, INetworkDefinition *network, int start_index, int end_index, std::vector<int>& output_index_vec) {

	for (int iter1 = start_index ; iter1 <= end_index; iter1++) {
		ILayer *layer = network->getLayer(iter1);
		
		for (int iter2 = 0; iter2 < layer->getNbOutputs(); iter2++) {
			ITensor *tensor = layer->getOutput(iter2);
			auto tensor_name = tensor->getName();

			if(checkTensorIsUsedInNextStages(device_id, network, end_index, iter1, tensor_name)) {
				output_index_vec.push_back(iter1);	
			}
		}
	}
}

void OnnxModel::fillInputs(int device_id, INetworkDefinition *network, int start_index, int end_index, std::vector<std::string>& input_name_vec) {
	std::set<ITensor *> output_set;
	for(int iter1 = 0 ; iter1 < start_index ; iter1++) {
		ILayer *layer = network->getLayer(iter1);
		for (int iter2 = 0; iter2 < layer->getNbOutputs(); iter2++) {
			ITensor *tensor = layer->getOutput(iter2);
			if(output_set.find(tensor) == output_set.end()) {
				output_set.insert(tensor);
			}
		}
	}

	for(int iter1 = start_index ; iter1 <= end_index ; iter1++) {
		ILayer *layer = network->getLayer(iter1);
		for(int iter2 = 0 ; iter2 < layer->getNbInputs() ; iter2++) {
			ITensor *tensor = layer->getInput(iter2);
			if(output_set.find(tensor) != output_set.end() || iter1 == 0) {
				input_name_vec.push_back(tensor->getName());
				output_set.erase(tensor);
			}
		}
	}

}

void OnnxModel::surgeonOnnxByPolygraphy(int device_id, INetworkDefinition *network, std::string model_name, std::string onnx_file_name, int start_index, int end_index) {
	int result = -1;
	std::string cmd = "polygraphy surgeon extract " + model_name + " -o " + onnx_file_name;
	std::string inputs = " --inputs ";
	std::string outputs = " --outputs ";

	std::vector<std::string> input_name_vec, output_name_vec;
	std::vector<int> output_index_vec;


	//getIOTensorNamesOfLayer(network, start_index, input_name_vec, true);
	fillInputs(device_id, network, start_index, end_index, input_name_vec);
	getOutputIndexOfStage(device_id, network, start_index, end_index, output_index_vec);
	std::cerr<<"output_index_vec size: "<<output_index_vec.size()<<std::endl;
	for(int output_index : output_index_vec) {
		std::cerr<<"\toutput_index: "<<output_index<<std::endl;
		getIOTensorNamesOfLayer(network, output_index, output_name_vec, false);
	}

	for(auto name :input_name_vec) {
		inputs.append(name + ":auto:auto ");	
	}
	for(auto name :output_name_vec) {
		outputs.append(name + ":auto ");	
	}

	cmd = cmd + inputs + outputs;
	std::cerr<<"cmd: "<<cmd<<std::endl;
	result = system(cmd.c_str());
	if(result == -1 || result == 127) {
		std::cerr<<"ERROR occurs at "<<__func__<<":"<<__LINE__<<std::endl;	
	}
}

void OnnxModel::separateOnnxFile(INetworkDefinition *network, std::string model_name, std::vector<std::string>& onnx_file_name_vec) {
	int device_num = config_data->instances.at(instance_id).device_num;
	int prev_cut_point = 0, curr_cut_point = 0;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		std::string onnx_file_name;

		getModelFileName(iter1, onnx_file_name, network, ".onnx");

		onnx_file_name_vec.push_back(onnx_file_name);
		if(iter1 > 0) {
			prev_cut_point = curr_cut_point + 1;
		}
		curr_cut_point = config_data->instances.at(instance_id).cut_points.at(iter1);
		curr_cut_point = std::min(curr_cut_point, network->getNbLayers()-1);

		if(fileExist(onnx_file_name) == false)  {
			surgeonOnnxByPolygraphy(iter1, network, model_name, onnx_file_name, prev_cut_point, curr_cut_point);
		}
	}
}

void OnnxModel::createEngineFromOnnxFile(int cur_iter, std::string onnx_file_name, IBuilder* &builder, INetworkDefinition* &network, IParser* &parser) {
	int device_num = config_data->instances.at(instance_id).device_num;
	int data_type = config_data->instances.at(instance_id).data_types.at(cur_iter);
	int device = config_data->instances.at(instance_id).devices.at(cur_iter);

	builder = createInferBuilder(logger);

	uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
	network =  (builder)->createNetworkV2(flag);

	parser = createParser(*network, logger);

	// TODO: onnx file path
	(parser)->parseFromFile(onnx_file_name.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < (parser)->getNbErrors(); ++i)
	{
		std::cout << "TENSORRT ONNX ERROR: "  << parser->getError(i)->desc() << std::endl;
	}

	if((parser)->getNbErrors() > 0) {
		FatalError("Onnx parsing failed");
	}

	int layer_num = network->getNbLayers();
	if(data_type == TYPE_INT8 && device == DEVICE_GPU) {
		for(int index = 0 ; index < layer_num ; index++) {
			ILayer *layer = network->getLayer(index);
			if(layer->getType() == nvinfer1::LayerType::kPOOLING) {
				IPoolingLayer *poolLayer = (IPoolingLayer *) layer;
				if(poolLayer->getPoolingType() == PoolingType::kMAX){
					layer->setPrecision( nvinfer1::DataType::kHALF);
				}
			}

			int output_num = layer->getNbOutputs();
			for(int out_index = 0; out_index < output_num ; out_index++) {
				ITensor *tensor = layer->getOutput(out_index);
				if(tensor != nullptr && tensor->isNetworkOutput()) {
					layer->setPrecision(nvinfer1::DataType::kHALF);
					break;
				}
			}
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

	if ((data_type == TYPE_FP16 || data_type == TYPE_INT8) &&  cur_iter > 0) {
		int input_num = network->getNbInputs();
		for (int  index = 0 ; index < input_num ; index++) {
			ITensor *tensor = network->getInput(index);
			tensor->setType(nvinfer1::DataType::kHALF);
		}
	}

	if ((data_type == TYPE_FP16 || data_type == TYPE_INT8) &&  device_num > 1 && cur_iter < device_num - 1) {
		int output_num = network->getNbOutputs();
		for (int  index = 0 ; index < output_num ; index++) {
			ITensor *tensor = network->getOutput(index);
			tensor->setType(nvinfer1::DataType::kHALF);
			std::cerr << "output type set to FP16: " << tensor->getName() << std::endl;
		}
	}
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


void OnnxModel::initializeModel() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int batch = config_data->instances.at(instance_id).batch;
	TensorRTNetwork *tensorrt_network = nullptr;
	IBuilder *builder = nullptr;
	INetworkDefinition *network = nullptr;

	int start_index = 0;

	tensorrt_network = dynamic_cast<TensorRTNetwork *>(app->createNetwork(&(config_data->instances.at(instance_id))));
	builder = tensorrt_network->builder;
	network = tensorrt_network->network;
	tensorrt_network->printNetwork();

	ITensor *tensor = network->getInput(0);
	Dims tensor_dim = tensor->getDimensions();
	
	total_input_size = 1;
	for(int iter1 = 0 ; iter1 < tensor_dim.nbDims ; iter1++) {
		total_input_size *= tensor_dim.d[iter1];
	}

	std::vector<std::string> onnx_file_name_vec;	
	separateOnnxFile(network, tensorrt_network->onnx_file_path, onnx_file_name_vec);

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		int cut_point = config_data->instances.at(instance_id).cut_points[iter1];
		int dla_core = config_data->instances.at(instance_id).dla_cores[iter1];
		int device = config_data->instances.at(instance_id).devices.at(iter1);
		int data_type = config_data->instances.at(instance_id).data_types.at(iter1);
		std::string plan_file_name;
		getModelFileName(iter1, plan_file_name, network, ".rt");

		if(fileExist(plan_file_name) == false)  {
			IBuilder *partial_builder;
			INetworkDefinition *partial_network;
			IParser *partial_parser;
			createEngineFromOnnxFile(iter1, onnx_file_name_vec[iter1], partial_builder, partial_network, partial_parser);

			IBuilderConfig* config = partial_builder->createBuilderConfig();
			config->setAvgTimingIterations(1);
			config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);
			config->setFlag(BuilderFlag::kDEBUG);
			ITimingCache *cache = nullptr;
			loadTimingCache(config, cache);
			config->setTimingCache(*cache, false);
			config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

			IOptimizationProfile* profile = partial_builder->createOptimizationProfile();	
			for(int iter2 = 0; iter2 < partial_network->getNbInputs(); iter2++) {
				ITensor *tensor = partial_network->getInput(iter2);
				Dims tensor_dim = tensor->getDimensions();
				// change batch size of a dynamic onnx model
				tensor_dim.d[0] = batch;

				profile->setDimensions(tensor->getName(), OptProfileSelector::kMIN, tensor_dim);
				profile->setDimensions(tensor->getName(), OptProfileSelector::kOPT, tensor_dim);
				profile->setDimensions(tensor->getName(), OptProfileSelector::kMAX, tensor_dim);
			}
			config->addOptimizationProfile(profile);
	

			// DLA options	
			if (device == DEVICE_DLA) {
				config->setFlag(BuilderFlag::kFP16);
				config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
				config->setDLACore(dla_core);
				config->setFlag(BuilderFlag::kGPU_FALLBACK);
				// config->setFlag(BuilderFlag::kSTRICT_TYPES);
				config->setFlag(BuilderFlag::kDIRECT_IO);
				config->setFlag(BuilderFlag::kREJECT_EMPTY_ALGORITHMS);
				config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM, 1U << 20);
				//config->setProfilingVerbosity( nvinfer1::ProfilingVerbosity::kDETAILED);
			}

			if(data_type == TYPE_FP16 && partial_builder->platformHasFastFp16()) {
				config->setFlag(BuilderFlag::kFP16);
			}
			else if(data_type == TYPE_INT8 && partial_builder->platformHasFastInt8()) {  	// int8 option
				if(partial_builder->platformHasFastFp16()) {
					config->setFlag(BuilderFlag::kFP16);
				}
				config->setFlag(BuilderFlag::kINT8);
				config->setInt8Calibrator(tensorrt_network->calibrator);
			}

			IHostMemory *serializedModel = partial_builder->buildSerializedNetwork(*partial_network, *config);
			assert(serializedModel != nullptr);

			serialize(plan_file_name.c_str(), serializedModel);
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
			ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream, size);
			assert(engine != nullptr);
			//auto inspector = std::unique_ptr<IEngineInspector>(engine->createEngineInspector());
			//std::cout << inspector->getLayerInformation(0, LayerInformationFormat::kJSON); // Print the information of the first layer in the engine.
			//std::cout << inspector->getEngineInformation(LayerInformationFormat::kJSON);
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


