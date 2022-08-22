#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <cctype>

#include <NvInfer.h>
#include <tkDNN/tkdnn.h>
#include <tkDNN/DarknetParser.h>

#include "cuda.h"

#include "model.h"
#include "variable.h"

Stage::Stage(ConfigData *config_data, int instance_id, int stage_id, int start_index, int end_index) {
	this->config_data = config_data;
	this->instance_id = instance_id;
	this->stage_id = stage_id;
	this->start_index = start_index;
	this->end_index = end_index;

	this->stream_num = config_data->instances.at(instance_id).stream_numbers[stage_id];
	this->buffer_num = config_data->instances.at(instance_id).buffer_num;

	this->input_size_map = std::map<std::pair<int, int>, int>();
	this->output_size_map = std::map<std::pair<int, int>, int>();
}

Stage::~Stage() {
	netRTs.clear();
	contexts.clear();
	streams.clear();
	events.clear();
}

void Stage::createExecutionContext() {
	for(int iter1 = 0; iter1 < stream_num; iter1++) {
		int size = netRTs.size();
		int index = size == 1 ? 0 : iter1 % DLA_NUM;

		nvinfer1::IExecutionContext *context = netRTs[index]->engineRT->createExecutionContext();
		assert(context);

		contexts.push_back(context);
	}
}

void Stage::setInputOutputLayerId(tk::dnn::Network *net) {
	input_size_map = tk::dnn::NetworkRT::getInputPair(net, start_index, end_index);
	output_size_map = tk::dnn::NetworkRT::getOutputPair(net, start_index, end_index);
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

void Stage::setBuffers(int buffer_id, std::map<std::pair<int, int>, void*> stream_buffers_map) {
	std::vector<void *> buffers;
	std::vector<int> input_indexes, output_indexes;
	
	for(auto iter = stream_buffers_map.begin(); iter != stream_buffers_map.end(); iter++) {
		auto pair_ids = iter->first;
		int src_id = pair_ids.first;
		int dst_id = pair_ids.second;

		if(dst_id >= start_index && dst_id <= end_index) {
			if(std::find(input_indexes.begin(), input_indexes.end(), src_id) == input_indexes.end()) {
				buffers.push_back(iter->second);
				input_indexes.push_back(src_id);
			}
		}
	}

	for(auto iter = stream_buffers_map.begin(); iter != stream_buffers_map.end(); iter++) {
		auto pair_ids = iter->first;
		int src_id = pair_ids.first;

		if(src_id >= start_index && src_id <= end_index) {
			if(std::find(output_indexes.begin(), output_indexes.end(), src_id) == output_indexes.end()) {
				buffers.push_back(iter->second);
				output_indexes.push_back(src_id);
			}
		}
	}

	stage_buffers.push_back(buffers);
}

void Stage::setSignals(int buffer_id, std::map<std::pair<int, int>, bool*> signals_map) {
	std::vector<bool *> input_signals;
	std::vector<bool *> output_signals;

	for(auto iter = signals_map.begin(); iter != signals_map.end(); iter++) {
		auto pair_ids = iter->first;
		int src_id = pair_ids.first;
		int dst_id = pair_ids.second;

		if(dst_id >= start_index && dst_id <= end_index) {
			input_signals.push_back(iter->second);
		}
		if(src_id >= start_index && src_id <= end_index) {
			output_signals.push_back(iter->second);
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

	for(unsigned int iter1 = 0; iter1 < netRTs.size(); iter1++) {
		delete netRTs[iter1];
	}
}

void Stage::getBindingsDataType() {
	std::cerr<<"stage_id: "<<stage_id<<std::endl;

    for(int iter = 0; iter < netRTs[0]->engineRT->getNbBindings(); iter++) {
		int type = (int)netRTs[0]->engineRT->getBindingDataType(iter);	
		int format = (int)netRTs[0]->engineRT->getBindingFormat(iter);	
		std::cerr<<"iter: "<<iter<<" type: "<<type<<std::endl;
		std::cerr<<"iter: "<<iter<<" binding format: "<<format<<std::endl;
    }
}
