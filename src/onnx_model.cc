#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cctype>

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

void OnnxModel::getModelFileName(int curr, std::string &plan_file_name, INetworkDefinition *network) {
	std::string model_dir = config_data->instances.at(instance_id).model_dir;
	std::string cut_points_name;
	std::string device_name;
	std::string data_type_name;
	std::string image_size_name;
	int device = config_data->instances.at(instance_id).devices.at(curr);
	int data_type = config_data->instances.at(instance_id).data_type;
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


	plan_file_name = model_dir + "/model_onnx_" + input_dim_name  + "_" + cut_points_name + "_" + device_name + "_" + data_type_name + ".rt";
	std::cerr<<"plan_file_name: "<<plan_file_name<<std::endl;
}



bool OnnxModel::serialize(const char *filename, nvinfer1::IHostMemory *ptr){
    std::ofstream p(filename, std::ios::binary);
    if (!p) {
		std::cerr << "Could not open plan output file" << std::endl;
		return false;
    }

    if(ptr == nullptr)
		std::cerr << "Can't serialize network" << std::endl;

    p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
    return true;
}


void OnnxModel::initializeModel() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int data_type = config_data->instances.at(instance_id).data_type;
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

	// TODO: polygraphy might be called here to make separate onnx_file

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		int cut_point = config_data->instances.at(instance_id).cut_points[iter1];
		int dla_core = config_data->instances.at(instance_id).dla_cores[iter1];
		int device = config_data->instances.at(instance_id).devices.at(iter1);
		std::string plan_file_name;

		// TODO: separate network definition for partial onnx and engine building is needed here

		getModelFileName(iter1, plan_file_name, network);
		if(fileExist(plan_file_name) == false)  {
			IBuilderConfig* config = builder->createBuilderConfig();
			config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);
			config->setFlag(BuilderFlag::kDEBUG);

			// DLA options	
			if (device == DEVICE_DLA) {
				config->setFlag(BuilderFlag::kFP16);
				config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
				config->setDLACore(dla_core);
				config->setFlag(BuilderFlag::kGPU_FALLBACK);
				config->setFlag(BuilderFlag::kSTRICT_TYPES);
			}

			if(data_type == TYPE_FP16 && builder->platformHasFastFp16()) {
				config->setFlag(BuilderFlag::kFP16);
			}
			else if(data_type == TYPE_INT8 && builder->platformHasFastInt8()) {  	// int8 option
				config->setFlag(BuilderFlag::kINT8);
				config->setInt8Calibrator(tensorrt_network->calibrator);
			}

			IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);

			serialize(plan_file_name.c_str(), serializedModel);

			//delete parser;
			//delete network;
			delete config;
			//delete builder;

			delete serializedModel;
			delete tensorrt_network->calibrator;
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
			stage->engines.push_back(engine);

			if (gieModelStream) delete [] gieModelStream;

			runtimes.push_back(runtime);
		}
		//setInputOutputLayerId(net, start_index, cut_point);
		stages.push_back(stage);

		start_index = cut_point + 1;

	}

	delete network;
	//delete config;
	delete builder;


	for(int iter1 = 0; iter1 < device_num; iter1++) {
		Stage *stage = stages[iter1];
		stage->createExecutionContext();
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


