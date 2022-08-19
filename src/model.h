#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <tkDNN/tkdnn.h>

#include "variable.h"
#include "config.h"
#include "stage.h"

#include "inference_application.h"



class Model {
	public:
		InputDim input_dim;
		bool letter_box;

		tk::dnn::Network *net;
		std::vector<Stage *> stages;
		std::vector<std::map<std::pair<int, int>, void*>> all_stream_buffers;
		std::vector<std::map<std::pair<int, int>, bool*>> all_signals;

		std::vector<std::vector<float *>> net_input_buffers;
		std::vector<std::vector<float *>> net_output_buffers;
		std::vector<std::vector<bool *>> net_input_signals;
		std::vector<std::vector<bool *>> net_output_signals;
		int network_output_number;

		Model(ConfigData *config_data, int instance_id, IInferenceApplication *app);
		~Model();
		void initializeModel();
		void finalizeModel();
		void initializeBuffers();
		void finalizeBuffers();
		bool checkInferenceDone(int device_id, int stream_id);
		void infer(int device_id, int stream_id, int buffer_id);
		void waitUntilInferenceDone(int device_id, int stream_id);
		void waitUntilInputConsumed(int device_id, int stream_id);
		bool checkInputConsumed(int device_id, int stream_id);

		bool isPreprocessingRunnable(int buffer_id);
		bool isPostprocessingRunnable(int buffer_id);
		void updateInputSignals(int buffer_id, bool value);
		void updateOutputSignals(int buffer_id, bool value);

	private:
		ConfigData *config_data;
		int instance_id;
		IInferenceApplication *app;

		void getModelFileName(int curr, std::string &plan_file_name);
		void setDataType();
		void setDevice(int curr);
		void setMaxBatchSize();
		void createCalibrationTable(std::string plan_file_name, int iter, int start_index, int end_index);
		void readFromCalibrationTable(std::string basic_calibration_table, int start_index, int end_index, std::string out_calib_table, int device);
		int getLayerNumberFromCalibrationKey(std::string key);
		void allocateStream();
		void allocateBuffer();
		void setBufferForStage();
		void deallocateBuffer();
		void deallocateStream();
		void* makeCUDAArray(int size);
		void allocateInputStreamBuffer(std::map<std::pair<int, int>, void*>& stream_buffers_map, std::vector<float *>& input_buffer, std::map<std::pair<int, int>, bool*>& signals_map, std::vector<bool*>& input_signal);
		void* getOutputBufferOfLayer(std::map<std::pair<int, int>, void*>& stream_buffers_map, int tsrc_id);
		void allocateStreamBuffer(std::map<std::pair<int, int>, int> size_map, std::map<std::pair<int, int>, void*>& stream_buffers_map, std::vector<float *>& output_buffer, std::map<std::pair<int, int>, bool*>& signals_map, std::vector<bool*>& output_signal); 
};

#endif
