#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <cctype>

#include <NvInfer.h>
//#include <tkDNN/tkdnn.h>
//#include <tkDNN/DarknetParser.h>

#include "cuda.h"

#include "model.h"
#include "stage.h"
#include "variable.h"

Stage::Stage(ConfigData *config_data, int instance_id, int stage_id, int start_index, int end_index) {
	this->config_data = config_data;
	this->instance_id = instance_id;
	this->stage_id = stage_id;
	this->start_index = start_index;
	this->end_index = end_index;

	this->stream_num = config_data->instances.at(instance_id).stream_numbers[stage_id];
	this->buffer_num = config_data->instances.at(instance_id).buffer_num;

	this->input_size_vec = std::vector<std::pair<std::string, nvinfer1::Dims>>();
	this->output_size_vec = std::vector<std::pair<std::string, nvinfer1::Dims>>();

	this->batch = this->config_data->instances.at(instance_id).batch;
}

Stage::~Stage() {
	engines.clear();
	contexts.clear();
	streams.clear();
	events.clear();
}

void Stage::createExecutionContext() {
	for(int iter1 = 0; iter1 < stream_num; iter1++) {
		int size = engines.size();
		int index = size == 1 ? 0 : iter1 % DLA_NUM;
		bool isImplicit = engines[index]->hasImplicitBatchDimension();

		nvinfer1::IExecutionContext *context = engines[index]->createExecutionContext();
		assert(context);

		binding_num = engines[index]->getNbIOTensors();
		for(int iter2 = 0; iter2 < binding_num; iter2++) {
			auto const& name = engines[index]->getIOTensorName(iter2);
			auto const& mode = engines[index]->getTensorIOMode(name);
			nvinfer1::Dims dims = engines[index]->getTensorShape(name);

			if(mode == nvinfer1::TensorIOMode::kINPUT) {
				if(!isImplicit) {
					context->setInputShape(name, dims);
				}
				if(iter1 == 0) {
					std::string _name(name);
					input_size_vec.push_back(std::pair<std::string, nvinfer1::Dims>(_name, dims));
				}
			}
			else {
				if(iter1 == 0) {
					std::string _name(name);
					output_size_vec.push_back(std::pair<std::string, nvinfer1::Dims>(_name, dims));
				}
			}
		}

		contexts.push_back(context);
	}
}

void Stage::allocateStream() {
	streams.clear();
	events.clear();

	for(int iter1 = 0; iter1 < stream_num; iter1++) {
		cudaStream_t stream;
		cudaEvent_t event;
		check_error(cudaStreamCreate(&stream));
		check_error(cudaEventCreate(&event));
		streams.push_back(stream);
		events.push_back(event);
	}
}

void Stage::deallocateStream() {
	for(int iter1 = 0; iter1 < stream_num; iter1++) {
		cudaStream_t stream = streams.back();
		cudaStreamDestroy(stream);
		streams.pop_back();

		cudaEvent_t event = events.back();
		cudaEventDestroy(event);
	}
	streams.clear();
	events.clear();
}

void Stage::setBuffers(int buffer_id, std::map<std::string, void*> stream_buffers_map) {
	std::vector<void *> buffers;

	for(auto iter = input_size_vec.begin(); iter != input_size_vec.end(); iter++) {
		std::string tensor_name = iter->first;

		auto iter2 = stream_buffers_map.find(tensor_name);
		if(iter2 != stream_buffers_map.end()) {
			void *space = iter2->second;
			buffers.push_back(space);
		}
	}

	for(auto iter = output_size_vec.begin(); iter != output_size_vec.end(); iter++) {
		std::string tensor_name = iter->first;

		auto iter2 = stream_buffers_map.find(tensor_name);
		if(iter2 != stream_buffers_map.end()) {
			void *space = iter2->second;
			buffers.push_back(space);
		}
	}

	stage_buffers.push_back(buffers);
}

void Stage::setSignals(int buffer_id, std::map<std::string, bool*> signals_map) {
	std::vector<bool *> input_signals;
	std::vector<bool *> output_signals;

	for(auto iter = input_size_vec.begin(); iter != input_size_vec.end(); iter++) {
		std::string tensor_name = iter->first;

		auto iter2 = signals_map.find(tensor_name);
		if(iter2 != signals_map.end()) {
			input_signals.push_back(iter2->second);
		}
	}

	for(auto iter = output_size_vec.begin(); iter != output_size_vec.end(); iter++) {
		std::string tensor_name = iter->first;

		auto iter2 = signals_map.find(tensor_name);
		if(iter2 != signals_map.end()) {
			output_signals.push_back(iter2->second);
		}
	}

	stage_input_signals.push_back(input_signals);
	stage_output_signals.push_back(output_signals);
}

bool Stage::isRunnable(int buffer_id) {
	std::vector<bool *> input_signals = stage_input_signals[buffer_id];
	std::vector<bool *> output_signals = stage_output_signals[buffer_id];

	for(unsigned int iter = 0; iter < input_signals.size(); iter++) {
		if(*(input_signals[iter]) == false)	
			return false;
	}

	for(unsigned int iter = 0; iter < output_signals.size(); iter++) {
		if(*(output_signals[iter]) == true)
			return false;	
	}

	return true;
}

void Stage::updateInputSignals(int buffer_id, bool value) {
	std::vector<bool *> input_signals = stage_input_signals[buffer_id];

	for(unsigned int iter = 0; iter < input_signals.size(); iter++) {
		*(input_signals[iter]) = value;	
	}
}

void Stage::updateOutputSignals(int buffer_id, bool value) {
	std::vector<bool *> output_signals = stage_output_signals[buffer_id];

	for(unsigned int iter = 0; iter < output_signals.size(); iter++) {
		*(output_signals[iter]) = value;
	}
}

void Stage::finalizeStage() {
	for(unsigned int iter1 = 0; iter1 < contexts.size(); iter1++) {
		nvinfer1::IExecutionContext *context = contexts[iter1];		
		delete context;
	}
}

void Stage::getBindingsDataType() {
    std::cerr<<"stage_id: "<<stage_id<<std::endl;

    for(int iter = 0; iter < engines[0]->getNbIOTensors(); iter++) {
        auto const& name = engines[0]->getIOTensorName(iter);
        int type = (int)engines[0]->getTensorDataType(name);
        int format = (int)engines[0]->getTensorFormat(name);
        std::cerr<<"iter: "<<iter<<" type: "<<type<<std::endl;
        std::cerr<<"iter: "<<iter<<" binding format: "<<format<<std::endl;
    }
}


