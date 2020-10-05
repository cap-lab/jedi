#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <cassert>

#include "variable.h"
#include "config.h"

typedef struct _YoloData {
	float *mask;
	int n_masks;
	float *bias;
} YoloData;

class Model {
	public:
		ConfigData *pConfigData;
		int instance_id;
		std::vector<int> start_bindings;
		int total_binding_num;
		std::vector<int> binding_size;
		int yolo_num;
		std::vector<YoloData> yolos;

		tk::dnn::Network *net;
		std::vector<tk::dnn::NetworkRT *> netRTs;
		std::vector<std::vector<nvinfer1::IExecutionContext *>> contexts;
		std::vector<std::vector<cudaStream_t>> streams;
		std::vector<bool> is_net_output;
		std::vector<void *> stream_buffers;
		std::vector<float *> input_buffers;
		std::vector<float *> output_buffers;

		Model(ConfigData *pConfigData, int instance_id);
		void getModelFileName(int curr, char *fileName);
		void setDevice(int curr);
		void setMaxBatchSize();
		void initializeModel();
		void setBindingsNum(int curr, int &input_binding_num, int &output_binding_num);
		void initializeBindingVariables();
		void setBufferIndexing();
		void allocateStream();
		void deallocateStream();
		void setStreamBuffer();
		void allocateBuffer();
		void deallocateBuffer();
		void initializeBuffers();
		void finalizeBuffers();
};

#endif
