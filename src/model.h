#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <vector>
#include <cassert>

#include <tkDNN/tkdnn.h>

#include "variable.h"
#include "config.h"

#include "inference_application.h"



class Model {
	public:
		ConfigData *config_data;
		int instance_id;
		std::vector<int> start_bindings;
		int total_binding_num;
		std::vector<int> binding_size;
		InputDim input_dim;
		bool letter_box;
		IInferenceApplication *app;

		tk::dnn::Network *net;
		std::vector<std::vector<tk::dnn::NetworkRT *>> netRTs;
		std::vector<std::vector<nvinfer1::IExecutionContext *>> contexts;
		std::vector<std::vector<cudaStream_t>> streams;
		std::vector<std::vector<cudaEvent_t>> events;

		std::vector<bool> is_net_output;
		std::vector<void *> stream_buffers;
		std::vector<float *> input_buffers;
		std::vector<float *> output_buffers;
		int network_output_number;

		Model(ConfigData *config_data, int instance_id, IInferenceApplication *app);
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
		bool checkInferenceDone(int device_id, int stream_id);
		void infer(int device_id, int stream_id, int buffer_id);
		void waitUntilInferenceDone(int device_id, int stream_id);
		void waitUntilInputConsumed(int device_id, int stream_id);
		bool checkInputConsumed(int device_id, int stream_id);
		void createCalibrationTable(std::string plan_file_name, int iter, int start_index, int end_index);
		void readFromCalibrationTable(std::string basic_calibration_table, int start_index, int end_index, std::string out_calib_table, int device);
		int getLayerNumberFromCalibrationKey(std::string key);


};

#endif
