#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <vector>
#include <cassert>

#include <tkDNN/tkdnn.h>

#include "variable.h"
#include "config.h"

typedef struct _YoloData {
	float *mask;
	int n_masks;
	float *bias;
} YoloData;

typedef struct _YoloValue {
	int height;
	int width;
	int channel;	
} YoloValue;

class Model {
	public:
		ConfigData *config_data;
		int instance_id;
		std::vector<int> start_bindings;
		int total_binding_num;
		std::vector<int> binding_size;
		int yolo_num;
		std::vector<YoloData> yolos;
		std::vector<YoloValue> yolo_values;
		InputDim input_dim;

		tk::dnn::Network *net;
		std::vector<tk::dnn::NetworkRT *> netRTs;
		std::vector<std::vector<nvinfer1::IExecutionContext *>> contexts;
		std::vector<std::vector<cudaStream_t>> streams;
		std::vector<bool> is_net_output;
		std::vector<void *> stream_buffers;
		std::vector<float *> input_buffers;
		std::vector<float *> output_buffers;

		Model(ConfigData *config_data, int instance_id);
		~Model();
		void getModelFileName(int curr, std::string &plan_file_name);
		void setDevice(int curr);
		void setMaxBatchSize();
		void setDataType();
		void initializeModel();
		void finalizeModel();
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
		bool checkInferenceDone(int device_id, int buffer_id);
		void infer(int device_id, int buffer_id);
		void waitUntilInferenceDone(int device_id, int buffer_id);
};

#endif
