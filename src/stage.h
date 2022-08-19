#ifndef STAGE_H_
#define STAGE_H_

#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <tkDNN/tkdnn.h>

#include "variable.h"
#include "config.h"


class Stage {
	public:
		std::vector<tk::dnn::NetworkRT *> netRTs;
		std::vector<nvinfer1::IExecutionContext *> contexts;
		std::vector<cudaStream_t> streams;
		std::vector<cudaEvent_t> events;

		std::vector<std::vector<void *>> stage_buffers;
		std::map<std::pair<int, int>, int> input_size_map;
		std::map<std::pair<int, int>, int> output_size_map;

		Stage(ConfigData *config_data, int instance_id, int stage_id, int start_index, int end_index);
		~Stage();

		void createExecutionContext();
		void setInputOutputLayerId(tk::dnn::Network *net);
		void allocateStream();
		void deallocateStream();
		void setBuffers(int buffer_id, std::map<std::pair<int, int>, void*> stream_buffers_map);
		void setSignals(int buffer_id, std::map<std::pair<int, int>, bool*> signals_map);
		bool isRunnable(int buffer_id);
		void updateInputSignals(int buffer_id, bool value);
		void updateOutputSignals(int buffer_id, bool value);
		void finalizeStage();
	
	private:
		ConfigData *config_data;
		int instance_id;
		int stage_id;
		int start_index;
		int end_index;
		int stream_num;
		int buffer_num;

		std::vector<std::vector<bool *>> stage_input_signals;
		std::vector<std::vector<bool *>> stage_output_signals;
};

#endif
