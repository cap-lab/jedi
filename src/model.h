#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <vector>
#include <cassert>

#include <tkDNN/tkdnn.h>

#include "variable.h"
#include "config.h"
#include "profiler.h"

typedef struct _YoloData {
	float *mask;
	int n_masks;
	float *bias;
	int new_coords;
	double nms_thresh;
	tk::dnn::Yolo::nmsKind_t nms_kind;
	int height;
	int width;
	int channel;	
} YoloData;


class Model {
	public:
		ConfigData *config_data;
		int instance_id;
		std::vector<int> start_bindings;
		int total_binding_num;
		std::vector<int> binding_size;
		std::vector<YoloData> yolos;
		InputDim input_dim;

		tk::dnn::Network *net;
		std::vector<std::vector<tk::dnn::NetworkRT *>> netRTs;
		std::vector<std::vector<nvinfer1::IExecutionContext *>> contexts;
		std::vector<std::vector<cudaStream_t>> streams;
		std::vector<std::vector<cudaEvent_t>> events;

		std::vector<bool> is_net_output;
		std::vector<void *> stream_buffers;
		std::vector<float *> input_buffers;
		std::vector<float *> output_buffers;
		std::vector<std::vector<long>> dla_profile_vec;
		Profiler profiler;

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
		void printProfile(std::string max_profile_file_name, std::string avg_profile_file_name);
		void createCalibrationTable(std::string plan_file_name, int iter, int start_index, int end_index);
		void readFromCalibrationTable(std::string basic_calibration_table, int start_index, int end_index, std::string out_calib_table);
		int getLayerNumberFromCalibrationKey(std::string key);
};

#endif
