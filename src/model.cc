#include <iostream>
#include <vector>
#include <cassert>
#include <cctype>

#include <NvInfer.h>

#include "cuda.h"

#include "model.h"
#include "variable.h"

NetworkModelRegistry g_NetworkModelRegistry = NetworkModelRegistry();


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

void* Model::makeCUDAArray(int size) {
	void *space;

	int data_type = this->config_data->instances.at(instance_id).data_type;

	if(data_type == TYPE_FP32) {
		 space = (void *)cuda_make_array(NULL, size);
	}
	else {
		space = (void *)cuda_make_array_16(NULL, size);
	}
	
	return space;
}

void Model::allocateIOStreamBuffer(std::vector<std::pair<std::string, nvinfer1::Dims>> size_vec, std::map<std::string, void*>& stream_buffers_map, std::vector<float *>& buffers, std::map<std::string, bool*>& signals_map, std::vector<bool*>& signals) {
	for(auto iter1 = size_vec.begin(); iter1 != size_vec.end(); iter1++) {
		std::string tensor_name = iter1->first;
		nvinfer1::Dims dims = iter1->second;
		int size = 1;
		void *space = nullptr;
		bool *signal = new bool(false);

		for(int iter2 = 0; iter2 < dims.nbDims; iter2++)
			size = size * dims.d[iter2];

		float *buf = cuda_make_array_host(size);
		cudaHostGetDevicePointer(&(space), buf, 0); 
		buffers.push_back(buf);
		stream_buffers_map.insert(std::make_pair(tensor_name, space));
		// fprintf(stderr, "[%s:%s:%d] tensor name: %s, space: %p, host space: %p\n", __FILE__, __func__, __LINE__, tensor_name.c_str(), space, buf);

		signals.push_back(signal);
		signals_map.insert(std::make_pair(tensor_name, signal));
	}   
}

void Model::allocateStreamBuffer(int stage_id, int is_input_size_map, std::vector<std::pair<std::string, nvinfer1::Dims>> size_vec, std::map<std::string, void*>& stream_buffers_map, std::map<std::string, bool*>& signals_map) {
	for(auto iter1 = size_vec.begin(); iter1 != size_vec.end(); iter1++) {
		std::string tensor_name = iter1->first;

		if(stream_buffers_map.find(tensor_name) == stream_buffers_map.end()) {
			// skip the first stage's input and the last stage's output
			if(stage_id == 0 && is_input_size_map)
				continue;
			if(stage_id == int(stages.size()-1) && !is_input_size_map)
				continue;

			nvinfer1::Dims dims = iter1->second;
			int size = 1;
			for(int iter2 = 0; iter2 < dims.nbDims; iter2++)
				size = size * dims.d[iter2];
			void *space = nullptr;
			bool *signal = new bool(false);

			space = makeCUDAArray(size);
			// fprintf(stderr, "[%s:%s:%d] tensor name: %s, space: %p\n", __FILE__, __func__, __LINE__, tensor_name.c_str(), space);

			stream_buffers_map.insert(std::make_pair(tensor_name, space));
			signals_map.insert(std::make_pair(tensor_name, signal));
		}
	}
}

void Model::allocateBuffer() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int buffer_id = 0; buffer_id < buffer_num; buffer_id++) {
		std::map<std::string, void*> stream_buffers_map;
		std::map<std::string, bool*> signals_map;
		std::vector<float*> input_buffer;
		std::vector<float*> output_buffer;
		std::vector<bool*> input_signal;
		std::vector<bool*> output_signal;

		allocateIOStreamBuffer(stages[0]->input_size_vec, stream_buffers_map, input_buffer, signals_map, input_signal);

		for(unsigned int stage_id = 0; stage_id < stages.size(); stage_id++) {
			Stage *stage = stages[stage_id];

			allocateStreamBuffer(stage_id, true, stage->input_size_vec, stream_buffers_map, signals_map);
			allocateStreamBuffer(stage_id, false, stage->output_size_vec, stream_buffers_map, signals_map);
		}

		allocateIOStreamBuffer(stages[stages.size()-1]->output_size_vec, stream_buffers_map, output_buffer, signals_map, output_signal);

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

		for(auto iter = stream_buffers_map.begin(); iter != stream_buffers_map.end(); iter++) {
			void *buffer = iter->second;
			if(buffer != nullptr)
				cudaFree(buffer);
		}

		for(auto iter = signals_map.begin(); iter != signals_map.end(); iter++) {
			delete iter->second;
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
		TensorAllocator *allocator = stage->tensor_allocators[buffer_id][stream_id][iter1];

		if(!isInput) {
			result = context->setOutputAllocator(name, allocator);
			assert(result);

			// fprintf(stderr, "[%s:%s:%d] tensor name: %s, space: %p, host space: %p\n", __FILE__, __func__, __LINE__, name, allocator->getBuf(), allocator->getHostBuf());
		}
		else {
			result = context->setTensorAddress(name, allocator->getBuf());	
			assert(result);

			// fprintf(stderr, "[%s:%s:%d] tensor name: %s, space: %p, host space: %p\n", __FILE__, __func__, __LINE__, name, allocator->getBuf(), allocator->getHostBuf());
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
	int batch = config_data->instances.at(instance_id).batch;
	bool enqueueSuccess = false;

	if(!stage->contexts[stream_id]->getEngine().hasImplicitBatchDimension()) {
		// stage->contexts[stream_id]->setOptimizationProfileAsync(0, stage->streams[stream_id]);
		// setBindingForContext(stage, stream_id, buffer_id);
		// enqueueSuccess = stage->contexts[stream_id]->enqueueV3(stage->streams[stream_id]);

		// setStreamBuffers(stage, stream_id, buffer_id);
		enqueueSuccess = stage->contexts[stream_id]->enqueueV2(&(stage->stage_buffers[buffer_id][0]), stage->streams[stream_id], &(stage->events[stream_id]));
	}
	else {
		enqueueSuccess = stage->contexts[stream_id]->enqueue(batch, &(stage->stage_buffers[buffer_id][0]), stage->streams[stream_id], &(stage->events[stream_id]));
		// enqueueSuccess = stage->contexts[stream_id]->execute(batch, &(stage->stage_buffers[buffer_id][0]));
	}

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



