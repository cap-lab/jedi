#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <cctype>

#include <NvInfer.h>

#include "cuda_jedi.h"

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
	this->device_num = config_data->instances.at(instance_id).device_num;
	this->data_type = config_data->instances.at(instance_id).data_types[stage_id];

	this->input_size_vec = std::vector<std::pair<std::string, nvinfer1::Dims>>();
	this->output_size_vec = std::vector<std::pair<std::string, nvinfer1::Dims>>();

	this->batch = this->config_data->instances.at(instance_id).batch;

	this->binding_num = 0;
	this->input_binding_num = 0;
	this->output_binding_num = 0;

	this->tensor_allocators = std::vector<std::vector<TensorAllocator *>>();
	for(int iter = 0; iter < buffer_num; iter++) {
		std::vector<TensorAllocator *> tensor_allocator;
		this->tensor_allocators.push_back(tensor_allocator);
	}
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
#if NV_TENSORRT_MAJOR > 9
		bool isImplicit = false;
#else
		bool isImplicit = engines[index]->hasImplicitBatchDimension();
#endif

		nvinfer1::IExecutionContext *context = engines[index]->createExecutionContext();
		assert(context);

		binding_num = engines[index]->getNbIOTensors();
		for(int iter2 = 0; iter2 < binding_num; iter2++) {
			auto const& name = engines[index]->getIOTensorName(iter2);
			auto const& mode = engines[index]->getTensorIOMode(name);

			if(mode == nvinfer1::TensorIOMode::kINPUT) {
				nvinfer1::Dims dims = engines[index]->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);

				if(!isImplicit) {
					// fprintf(stderr, "[%s:%s:%d] tensor name: %s dims: [%dx%dx%dx%d]\n", __FILE__, __func__, __LINE__, name, dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
					context->setInputShape(name, dims);
				}
				if(iter1 == 0) {
					std::string _name(name);
					input_size_map[_name] = dims;
					input_type_map[_name] = context->getEngine().getTensorDataType(name);
					input_size_vec.push_back(std::pair<std::string, nvinfer1::Dims>(_name, dims));
					input_binding_num++;
				}
			}
			else {
				nvinfer1::Dims dims = engines[index]->getTensorShape(name);

				if(iter1 == 0) {
					std::string _name(name);
					output_size_map[_name] = dims;
					output_type_map[_name] = context->getEngine().getTensorDataType(name);
					output_size_vec.push_back(std::pair<std::string, nvinfer1::Dims>(_name, dims));
					output_binding_num++;
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
		check_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
		events.pop_back();
	}
	streams.clear();
	events.clear();
}

void Stage::setBuffers(int buffer_id, std::map<std::string, void*> stream_buffers_map) {
	std::vector<void *> buffers;

//	for(int iter = 0; iter < binding_num; iter++)
//		buffers.push_back(nullptr);

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

uint64_t Stage::getSizeByTensorName(bool isInput, std::string name) {
	nvinfer1::Dims dims;
	uint64_t size = 1;

	if(isInput) {
		for(auto iter = input_size_map.begin(); iter != input_size_map.end(); iter++) {
			if(name.compare(iter->first) == 0) {
				dims = iter->second;

				for(int iter2 = 0; iter2 < dims.nbDims; iter2++)
					size = size * dims.d[iter2];
				size = size * getDataTypeSize(input_type_map.find(iter->first)->second);
				break;
			}
		}
	}
	else {
		for(auto iter = output_size_map.begin(); iter != output_size_map.end(); iter++) {
			if(name.compare(iter->first) == 0) {
				dims = iter->second;

				for(int iter2 = 0; iter2 < dims.nbDims; iter2++)
					size = size * dims.d[iter2];
				size = size * getDataTypeSize(output_type_map.find(iter->first)->second);
				break;
			}
		}
	}

	return size;
}

void Stage::setTensorAllocators(int buffer_id, std::map<std::string, void*> stream_buffers_map, std::vector<void *> input_buffers, std::vector<void *> output_buffers) {
		auto context = contexts[0];
		std::vector<TensorAllocator *> tensor_allocator = tensor_allocators[buffer_id];
		//nvinfer1::DataType dtype = (data_type == TYPE_FP32) ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;

		for(int iter2 = 0; iter2 < binding_num; iter2++) {
			auto const& name = context->getEngine().getIOTensorName(iter2);
			auto const& mode = context->getEngine().getTensorIOMode(name);
			bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT) ? true : false;
			std::string _name(name);

			bool is_host_allocated = false;
			if( (stage_id == 0 && is_input) || ((stage_id == (device_num-1)) && !is_input) ) 
				is_host_allocated = true;
			uint64_t size = getSizeByTensorName(is_input, _name);
			void *buf = stream_buffers_map[_name];
			void *host_buf = !is_host_allocated ? nullptr : (is_input ? input_buffers[iter2] : output_buffers[iter2-input_binding_num]);

			if(size % ALIGNMENT != 0) {
				size = (size / ALIGNMENT + 1) * ALIGNMENT;
			}

			// fprintf(stderr, "tensor name: %s, is_host_allocated: %d, host_buf: %p, size: %ld\n", name, is_host_allocated, host_buf, size);
			TensorAllocator *allocator = new TensorAllocator(is_host_allocated, buf, host_buf, size, context->getEngine().getTensorDataType(name));

			tensor_allocator.push_back(allocator);
		}
		this->tensor_allocators[buffer_id] = tensor_allocator;
}

void Stage::setSignals(int buffer_id, std::map<std::string, BufferSignal*> signals_map) {
	std::vector<BufferSignal *> input_signals;
	std::vector<BufferSignal *> output_signals;

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
	std::vector<BufferSignal *> input_signals = stage_input_signals[buffer_id];
	std::vector<BufferSignal *> output_signals = stage_output_signals[buffer_id];

	for(unsigned int iter = 0; iter < input_signals.size(); iter++) {
		if(input_signals[iter]->isSignalSet() == false)
			return false;
	}

	for(unsigned int iter = 0; iter < output_signals.size(); iter++) {
		if(output_signals[iter]->isSignalSet() == true)
			return false;	
	}

	return true;
}

void Stage::updateInputSignals(int buffer_id, bool value) {
	std::vector<BufferSignal *> input_signals = stage_input_signals[buffer_id];

	for(unsigned int iter = 0; iter < input_signals.size(); iter++) {
		input_signals[iter]->updateSignal(value);
	}
}

void Stage::updateOutputSignals(int buffer_id, bool value) {
	std::vector<BufferSignal *> output_signals = stage_output_signals[buffer_id];

	for(unsigned int iter = 0; iter < output_signals.size(); iter++) {
		output_signals[iter]->updateSignal(value);
	}
}

void Stage::finalizeStage() {
	for(unsigned int iter1 = 0; iter1 < contexts.size(); iter1++) {
		nvinfer1::IExecutionContext *context = contexts[iter1];		
		delete context;
	}

	for(unsigned int iter1 = 0; iter1 < tensor_allocators.size(); iter1++) {
		for(unsigned int iter2 = 0; iter2 < tensor_allocators[iter1].size(); iter2++) {
				// buffers are deallocated in the class model
				delete tensor_allocators[iter1][iter2];
		}
	}

	for(unsigned int iter1 = 0; iter1 < engines.size(); iter1++) {
		delete engines[iter1];
	}

	for(unsigned int iter1 = 0; iter1 < instances.size(); iter1++) {
#ifdef STRING_PER_BUFFER
		cudaGraphExecDestroy(instances[iter1]);
#else
		for(unsigned int iter2 = 0; iter2 < instances[iter1].size(); iter2++) {
			cudaGraphExecDestroy(instances[iter1][iter2]);
		}
#endif
	}
}

void Stage::initializeCudaGraphs(std::vector<std::map<std::string, void*>> all_stream_buffers) {

#ifdef STRING_PER_BUFFER
	cudaStream_t stream;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	for(unsigned int iter1 = 0; iter1 < tensor_allocators.size(); iter1++) {
		int stream_index = iter1 % stream_num;
		auto context = this->contexts[stream_index];
		int binding_num = context->getEngine().getNbIOTensors();
		cudaGraph_t graph;
		cudaGraphExec_t instance;

		for(int iter2 = 0; iter2 < binding_num; iter2++) {
			auto const& name = context->getEngine().getIOTensorName(iter2);
			auto const& mode = context->getEngine().getTensorIOMode(name);
			bool isInput = (mode == nvinfer1::TensorIOMode::kINPUT) ? true : false;
			bool result = false;
			std::string _name(name);
			TensorAllocator *allocator = this->tensor_allocators[iter1][iter2];

			if(!isInput) {
				result = context->setOutputAllocator(name, allocator);
				assert(result);
			}
			else {
				result = context->setTensorAddress(name, allocator->getBuf());
				assert(result);
			}

			if(allocator->getIsReallocated()) {
				all_stream_buffers[iter1][_name] = allocator->getBuf();
			}
		}

		check_error(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
		context->enqueueV3(stream);
		check_error(cudaStreamEndCapture(stream, &graph));
		check_error(cudaGraphInstantiate(&instance, graph, 0));
		check_error(cudaGraphDestroy(graph));
		//stream_instances.push_back(instance);
		this->instances.push_back(instance);
	}

	cudaStreamDestroy(stream);
#else
	cudaStream_t stream;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	for(int iter = 0 ; iter < stream_num ; iter++) {
		auto context = this->contexts[iter];
		std::vector<cudaGraphExec_t> stream_instances;

		for(unsigned int iter1 = 0; iter1 < tensor_allocators.size(); iter1++) {
			int binding_num = context->getEngine().getNbIOTensors();
			cudaGraph_t graph;
			cudaGraphExec_t instance;

			for(int iter2 = 0; iter2 < binding_num; iter2++) {
				auto const& name = context->getEngine().getIOTensorName(iter2);
				auto const& mode = context->getEngine().getTensorIOMode(name);
				bool isInput = (mode == nvinfer1::TensorIOMode::kINPUT) ? true : false;
				bool result = false;
				std::string _name(name);
				TensorAllocator *allocator = this->tensor_allocators[iter1][iter2];

				if(!isInput) {
					result = context->setOutputAllocator(name, allocator);
					assert(result);
				}
				else {
					result = context->setTensorAddress(name, allocator->getBuf());
					assert(result);
				}

				if(allocator->getIsReallocated()) {
					all_stream_buffers[iter1][_name] = allocator->getBuf();
				}
			}

			check_error(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
			context->enqueueV3(stream);
			check_error(cudaStreamEndCapture(stream, &graph));
			check_error(cudaGraphInstantiate(&instance, graph, 0));
			check_error(cudaGraphDestroy(graph));
			stream_instances.push_back(instance);
		}
		this->instances.push_back(stream_instances);
	}
	cudaStreamDestroy(stream);
#endif
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


