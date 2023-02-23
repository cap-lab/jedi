#ifndef STAGE_H_
#define STAGE_H_

#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <tuple>

#include <NvInfer.h>

//#include <tkDNN/tkdnn.h>

#include "variable.h"
#include "config.h"

#include "cuda.h"
#include "binding.h"

class Stage {
	public:
		std::vector<nvinfer1::ICudaEngine *> engines;
		std::vector<nvinfer1::IExecutionContext *> contexts;
		std::vector<cudaStream_t> streams;
		std::vector<cudaEvent_t> events;

		std::vector<TensorAllocator *> tensor_allocators;

		std::vector<std::vector<void *>> stage_buffers;
		std::vector<std::pair<std::string, nvinfer1::Dims>> input_size_vec;
		std::vector<std::pair<std::string, nvinfer1::Dims>> output_size_vec;

		int batch;
		int binding_num;

		Stage(ConfigData *config_data, int instance_id, int stage_id, int start_index, int end_index);
		~Stage();

		void createExecutionContext();
		void allocateStream();
		void deallocateStream();
		void setBuffers(int buffer_id, std::map<std::string, void*> stream_buffers_map);
		uint64_t getSizeByTensorName(bool isInput, std::string name);
		void setTensorAllocators(int buffer_id, std::map<std::string, void*> stream_buffers_map, std::vector<float *> input_buffers, std::vector<float *> output_buffers);
		void setSignals(int buffer_id, std::map<std::string, bool*> signals_map);
		bool isRunnable(int buffer_id);
		void updateInputSignals(int buffer_id, bool value);
		void updateOutputSignals(int buffer_id, bool value);
		void finalizeStage();
		void getBindingsDataType();
	
	private:
		ConfigData *config_data;
		int instance_id;
		int stage_id;
		int start_index;
		int end_index;
		int stream_num;
		int buffer_num;
		int device_num;

		std::vector<std::vector<bool *>> stage_input_signals;
		std::vector<std::vector<bool *>> stage_output_signals;
};

#endif
