#include <iostream>
#include <vector>
#include <cassert>
#include <cctype>
#include <sstream>
#include <set>

#include <NvInfer.h>

#include "cuda_jedi.h"

#include "model.h"
#include "variable.h"

NetworkModelRegistry g_NetworkModelRegistry = NetworkModelRegistry();

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

static int getAllocationSizeByDims(nvinfer1::Dims dims) {
	int size = 1;
	for(int iter2 = 0; iter2 < dims.nbDims; iter2++)
		size = size * dims.d[iter2];

	if(size % ALIGNMENT != 0) {
		size = (size / ALIGNMENT + 1) * ALIGNMENT;
	}
	return size;
}

Model::Model(ConfigData *config_data, int instance_id, IInferenceApplication *app) {
	this->config_data = config_data;
	this->instance_id = instance_id;
	this->network_output_number = 0;
	this->app = app;
}

Model::~Model() {
	for(unsigned int iter = 0; iter < stages.size(); iter++) {
		Stage *stage = stages[iter];
		delete stage;
	}
	stages.clear();
}

void Model::allocateStream() {
	for(unsigned int iter1 = 0; iter1 < stages.size(); iter1++) {
		Stage *stage = stages[iter1];
		stage->allocateStream();
	}
}

void Model::deallocateStream() {
	for(unsigned int iter1 = 0; iter1 < stages.size(); iter1++) {
		Stage *stage = stages[iter1];
		stage->deallocateStream();
	}
}

void Model::allocateIOStreamBuffer(std::vector<std::pair<std::string, nvinfer1::Dims>> size_vec, std::map<std::string, nvinfer1::DataType> type_map, std::map<std::string, void*>& stream_buffers_map, std::vector<void *>& buffers, std::map<std::string, bool*>& signals_map, std::vector<bool*>& signals) {
	for(auto iter = size_vec.begin(); iter != size_vec.end(); iter++) {
		std::string tensor_name = iter->first;
		nvinfer1::Dims dims = iter->second;
		int size = getAllocationSizeByDims(dims);
		void *space = nullptr;
		bool *signal = new bool(false);

		void *buf = cuda_make_generic_array_host(size, getDataTypeSize(type_map.find(iter->first)->second));
		cudaHostGetDevicePointer((void **) &(space), buf, 0); 
		buffers.push_back(buf);

		stream_buffers_map.insert(std::make_pair(tensor_name, space));
		// fprintf(stderr, "[%s:%s:%d] tensor name: %s, space: %p, host space: %p, size: %d\n", __FILE__, __func__, __LINE__, tensor_name.c_str(), space, buf, size);

		signals.push_back(signal);
		signals_map.insert(std::make_pair(tensor_name, signal));
	}
}


void Model::allocateStreamBuffer(int stage_id, int is_input_size_map, std::map<std::string, nvinfer1::Dims> size_map, std::map<std::string, nvinfer1::DataType> type_map, std::map<std::string, void*>& stream_buffers_map, std::map<std::string, bool*>& signals_map) {

	// skip the first stage's input and the last stage's output
	if(stage_id == 0 && is_input_size_map)
		return;
	if(stage_id == int(stages.size()-1) && !is_input_size_map)
		return;

	for(auto iter = size_map.begin(); iter != size_map.end(); iter++) {
		std::string tensor_name = iter->first;

		if(stream_buffers_map.find(tensor_name) == stream_buffers_map.end()) {
			nvinfer1::Dims dims = iter->second;
			int size = getAllocationSizeByDims(dims);
			void *space = nullptr;
			bool *signal = new bool(false);

			space = cuda_make_generic_array(nullptr, size, getDataTypeSize(type_map.find(iter->first)->second));
			// fprintf(stderr, "[%s:%s:%d] tensor name: %s, space: %p\n", __FILE__, __func__, __LINE__, tensor_name.c_str(), space);
			stream_buffers_map.insert(std::make_pair(tensor_name, space));
			signals_map.insert(std::make_pair(tensor_name, signal));
		}
	}
}

static bool isPrefixRelation(const std::string& str1, const std::string& str2) {
	std::string prefix;
	std::string main_str;

	if (str1.length() < str2.length()) {
		prefix = str1;
		main_str = str2;
	} else {
		prefix = str2;
		main_str = str1;
	}

	return main_str.compare(0, prefix.size(), prefix) == 0;
}

void Model::allocateMissingStreamBuffer(int stage_id, int is_input_size_map, std::map<std::string, nvinfer1::Dims> input_size_map, std::map<std::string, nvinfer1::DataType> input_type_map, std::map<std::string, void*>& stream_buffers_map, std::map<std::string, bool*>& signals_map) {
	// skip the first stage's input and the last stage's output
	if(stage_id == 0 && is_input_size_map)
		return;
	if(stage_id == int(stages.size()-1) && !is_input_size_map)
		return;

	for(auto iter = input_size_map.begin(); iter != input_size_map.end(); iter++) {
		std::string tensor_name = iter->first;

		if(stream_buffers_map.find(tensor_name) == stream_buffers_map.end()) {
			nvinfer1::Dims dims = iter->second;
			int size = getAllocationSizeByDims(dims);
			void *space = nullptr;
			bool *signal = nullptr;

			for (auto iter2 = stream_buffers_map.begin() ; iter2 != stream_buffers_map.end() ; iter2++) {
				if (input_size_map.find(iter2->first) == input_size_map.end() && isPrefixRelation(tensor_name, iter2->first)) {
					for(unsigned int prev_stage_id = 0; prev_stage_id < stage_id; prev_stage_id++) {
						Stage *stage = stages[prev_stage_id];
						if (stage->output_size_map.find(iter2->first) != stage->output_size_map.end()) {
							if(size == getAllocationSizeByDims(stage->output_size_map[iter2->first]) && input_type_map[iter->first] == stage->output_type_map[iter2->first]) {
								space = stream_buffers_map.find(iter2->first)->second;
								signal = signals_map.find(iter2->first)->second;
								break;
							}
						}
					}

					if(space != nullptr && signal != nullptr)
						break;
				}
			}

			if (space == nullptr || signal == nullptr) {
				FatalError("invalid input tensor name: " + tensor_name);
			}

			stream_buffers_map[tensor_name] = space;
			signals_map[tensor_name] = signal;
		}
	}
}

void Model::allocateBuffer() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int buffer_id = 0; buffer_id < buffer_num; buffer_id++) {
		std::map<std::string, void*> stream_buffers_map;
		std::map<std::string, bool*> signals_map;
		std::vector<void*> input_buffer;
		std::vector<void*> output_buffer;
		std::vector<bool*> input_signal;
		std::vector<bool*> output_signal;

		allocateIOStreamBuffer(stages[0]->input_size_vec, stages[0]->input_type_map, stream_buffers_map, input_buffer, signals_map, input_signal);

		for(unsigned int stage_id = 0; stage_id < stages.size(); stage_id++) {
			Stage *stage = stages[stage_id];

			allocateStreamBuffer(stage_id, false, stage->output_size_map, stage->output_type_map, stream_buffers_map, signals_map);
			allocateMissingStreamBuffer(stage_id, true, stage->input_size_map, stage->input_type_map, stream_buffers_map, signals_map);
		}

		allocateIOStreamBuffer(stages[stages.size()-1]->output_size_vec, stages[stages.size()-1]->output_type_map, stream_buffers_map, output_buffer, signals_map, output_signal);

		net_input_buffers.push_back(input_buffer);
		net_output_buffers.push_back(output_buffer);
		all_stream_buffers.push_back(stream_buffers_map);

		net_input_signals.push_back(input_signal);
		net_output_signals.push_back(output_signal);
		all_signals.push_back(signals_map);

		this->network_output_number = output_buffer.size();
	}
}

void Model::deallocateBuffer() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	if (net_input_buffers.size() == 0 && net_output_buffers.size() == 0) {
		return;
	}

	for(int buffer_id = 0; buffer_id < buffer_num; buffer_id++) {
		auto input_buffer = net_input_buffers[buffer_id];
		auto output_buffer = net_output_buffers[buffer_id];

		for(auto iter = input_buffer.begin(); iter != input_buffer.end(); iter++) {
			void *buffer = *iter;
			if(buffer != nullptr)
				cudaFreeHost(buffer);
		}

		for(auto iter = output_buffer.begin(); iter != output_buffer.end(); iter++) {
			void *buffer = *iter;
			if(buffer != nullptr)
				cudaFreeHost(buffer);
		}

		auto stream_buffers_map = all_stream_buffers[buffer_id];
		auto signals_map = all_signals[buffer_id];
		std::set<void *> buffer_removed;

		for(auto iter = stream_buffers_map.begin(); iter != stream_buffers_map.end(); iter++) {
			void *buffer = iter->second;

			if(buffer_removed.find(buffer) == buffer_removed.end() && buffer != nullptr) {
				cudaFree(buffer);
				buffer_removed.insert(buffer);
			}
		}

		std::set<void *> signal_removed;

		for(auto iter = signals_map.begin(); iter != signals_map.end(); iter++) {
			if (signal_removed.find(iter->second) == signal_removed.end()) {
				delete iter->second;
				signal_removed.insert(iter->second);
			}
		}
	}
}

void Model::setBufferForStage() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int device_num = config_data->instances.at(instance_id).device_num;

	for(int iter1 = 0; iter1 < buffer_num; iter1++) {
		for(int iter2 = 0; iter2 < device_num; iter2++) {
			Stage *stage = stages[iter2];
			stage->setBuffers(iter1, all_stream_buffers[iter1]);
			stage->setSignals(iter1, all_signals[iter1]);
			stage->setTensorAllocators(iter1, all_stream_buffers[iter1], net_input_buffers[iter1], net_output_buffers[iter1]);
		}
	}
}

bool Model::isPreprocessingRunnable(int buffer_id) {
	std::vector<bool*> input_signal = net_input_signals[buffer_id];

	for(unsigned int iter = 0; iter < input_signal.size(); iter++) {
		if(*(input_signal[iter]) == true)	
			return false;
	}

	return true;
}

bool Model::isPostprocessingRunnable(int buffer_id) {
	std::vector<bool*> output_signal = net_output_signals[buffer_id];

	for(unsigned int iter = 0; iter < output_signal.size(); iter++) {
		if(*(output_signal[iter]) == false)	
			return false;
	}

	return true;
}

void Model::updateInputSignals(int buffer_id, bool value) {
	std::vector<bool*> input_signal = net_input_signals[buffer_id];

	for(unsigned int iter = 0; iter < input_signal.size(); iter++) {
		*(input_signal[iter]) = value;	
	}
}

void Model::updateOutputSignals(int buffer_id, bool value) {
	std::vector<bool*> output_signal = net_output_signals[buffer_id];

	for(unsigned int iter = 0; iter < output_signal.size(); iter++) {
		*(output_signal[iter]) = value;	
	}
}

void Model::initializeBuffers() {
	allocateStream();
	allocateBuffer();
	setBufferForStage();
}

void Model::finalizeBuffers() {
	deallocateBuffer();
	deallocateStream();
}

bool Model::checkInputConsumed(int device_id, int stream_id) {
	Stage *stage = stages[device_id];
	cudaError_t error = cudaEventQuery(stage->events[stream_id]);

	if(error == cudaSuccess)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void Model::initializeStreams(int device_id) {
	Stage *stage = stages[device_id];

	for(int iter = 0 ; iter < stage->streams.size() ; iter++) {
		cudaStreamSynchronize(stage->streams[iter]);
	}
}

bool Model::checkInferenceDone(int device_id, int stream_id) {
	Stage *stage = stages[device_id];
	cudaError_t error = cudaStreamQuery(stage->streams[stream_id]);	

	if(error == cudaSuccess)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void Model::setBindingForContext(Stage *stage, int stream_id, int buffer_id) {
	auto context = stage->contexts[stream_id];
	int binding_num = context->getEngine().getNbIOTensors();
	
	for(int iter1 = 0; iter1 < binding_num; iter1++) {
		auto const& name = context->getEngine().getIOTensorName(iter1);
		auto const& mode = context->getEngine().getTensorIOMode(name);
		bool isInput = (mode == nvinfer1::TensorIOMode::kINPUT) ? true : false;
		bool result = false;
		std::string _name(name);
		TensorAllocator *allocator = stage->tensor_allocators[buffer_id][iter1];

		if(!isInput) {
			result = context->setOutputAllocator(name, allocator);
			assert(result);

			// fprintf(stderr, "[%s:%s:%d] tensor name: %s, space: %p, host space: %p, buffer_id: %d, stream_id: %d, iter1: %d, size: %lu\n", __FILE__, __func__, __LINE__, name, allocator->getBuf(), allocator->getHostBuf(), buffer_id, stream_id, iter1, allocator->getSize());
		}
		else {
			result = context->setTensorAddress(name, allocator->getBuf());	
			assert(result);

			// fprintf(stderr, "[%s:%s:%d] tensor name: %s, space: %p, host space: %p, buffer_id: %d, stream_id: %d, iter1: %d, size: %lu\n", __FILE__, __func__, __LINE__, name, allocator->getBuf(), allocator->getHostBuf(), buffer_id, stream_id, iter1, allocator->getSize());
		}		

		if(allocator->getIsReallocated()) {
			all_stream_buffers[buffer_id][_name] = allocator->getBuf();
		}
	}
}

void Model::setStreamBuffers(Stage *stage, int stream_id, int buffer_id) {
	auto context = stage->contexts[stream_id];
	for(int iter = 0; iter < stage->binding_num; iter++) {
		auto const& name = context->getEngine().getIOTensorName(iter);
		std::string _name(name);
		stage->stage_buffers[buffer_id][iter] = all_stream_buffers[buffer_id][_name];
	}
}

void Model::infer(int device_id, int stream_id, int buffer_id) {
	Stage *stage = stages[device_id];
	bool enqueueSuccess = false;

#if NV_TENSORRT_MAJOR > 8
	stage->contexts[stream_id]->setOptimizationProfileAsync(0, stage->streams[stream_id]);
	setBindingForContext(stage, stream_id, buffer_id);
	enqueueSuccess = stage->contexts[stream_id]->enqueueV3(stage->streams[stream_id]);
#else
	if(!stage->contexts[stream_id]->getEngine().hasImplicitBatchDimension()) {
		stage->contexts[stream_id]->setOptimizationProfileAsync(0, stage->streams[stream_id]);
		setBindingForContext(stage, stream_id, buffer_id);
		enqueueSuccess = stage->contexts[stream_id]->enqueueV3(stage->streams[stream_id]);

		// setStreamBuffers(stage, stream_id, buffer_id);
		//enqueueSuccess = stage->contexts[stream_id]->enqueueV2(&(stage->stage_buffers[buffer_id][0]), stage->streams[stream_id], &(stage->events[stream_id]));
	}
	else {
		int batch = config_data->instances.at(instance_id).batch;
		enqueueSuccess = stage->contexts[stream_id]->enqueue(batch, &(stage->stage_buffers[buffer_id][0]), stage->streams[stream_id], &(stage->events[stream_id]));
		// enqueueSuccess = stage->contexts[stream_id]->execute(batch, &(stage->stage_buffers[buffer_id][0]));
	}
#endif

	if(enqueueSuccess == false)
	{
		printf("enqueue error happened: %d, %d\n", device_id, buffer_id);
		exit_flag = true;
	}
}

void Model::waitUntilInputConsumed(int device_id, int stream_id) {
	Stage *stage = stages[device_id];
	cudaError_t error;

	error = cudaEventSynchronize(stage->events[stream_id]);
	if(error != cudaSuccess)
	{
		printf("error happened in synchronize: %d, %d: %d\n", device_id, stream_id, error);
		exit_flag = true;
	}
}

void Model::waitUntilInferenceDone(int device_id, int stream_id) {
	Stage *stage = stages[device_id];
	cudaError_t error;

	error = cudaStreamSynchronize(stage->streams[stream_id]);
	if(error != cudaSuccess)
	{
		printf("error happened in synchronize: %d, %d: %d\n", device_id, stream_id, error);
		exit_flag = true;
	}
}



