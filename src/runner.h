#ifndef RUNNER_H_
#define RUNNER_H_

#include <iostream>
#include <vector>
#include <thread>

#include <tkDNN/tkdnn.h>

#include "config.h"
#include "variable.h"
#include "model.h"
#include "coco.h"

#include "inference_application.h"
#include "yolo_application.h"

#define PRE_LOCK 0
#define INFER_LOCK 1
#define POST_LOCK 2
#define LOCK_NUM 3

class Runner {
	public:
		Runner(std::string config_file_name, int width, int height, int channel, int step);
		~Runner();
		void init();
		void run(char *data);
		void run(char *data, char *result_file_name);
		void setThreadCores(int pre_thread_core, int post_thread_core);
		void wrapup();
		void saveProfileResults(char *max_filename, char *avg_filename, char *min_filename);

		void setInputData();
		void postProcess();

		int device_num = 0;
		float *input_buffer;
		std::vector<Model *> models;
		std::vector<int> signals;
		bool exit_flag = false;
		int pre_thread_core = 0;
		int post_thread_core = 0;

	private:
		void generateModels();
		void initializePreAndPostprocessing();
		void finalizeData();
		void runThreads();

		ConfigData *config_data;
		std::vector<IInferenceApplication *> apps;
		int width = 0;
		int height = 0;
		int channel = 3;
		int step = 0;
		char *image_data;	
		char *result_file_name;
		std::vector<std::thread> threads;
};

#endif
