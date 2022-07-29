#ifndef RUNNER2_H_
#define RUNNER2_H_

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

class Runner2 {
	public:
		Runner2(std::string config_file_name, int width, int height, int channel, int step);
		~Runner2();
		void init();
		void loadImage(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input); 
		void wrapup();
		void setInputData(float *input_buffer);
		void setInputData2(char *image_data);
		void runInference();
		void getOutputData(float **output_buffers);
		void doPostProcessing(char *input_data, char *result_file_name);
		void doPostProcessing(float **output_data, char *input_data, char *result_file_name);
		void saveProfileResults(char *max_filename, char *avg_filename, char *min_filename);
		void setThreadCores(int core_id);
		void saveResults(char *result_file_name);

		int device_num = 0;
		int input_size = 0;
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

		ConfigData *config_data;
		std::vector<IInferenceApplication *> apps;
		int width = 0;
		int height = 0;
		int channel = 3;
		int step = 0;
		float **output_pointers;
		std::vector<std::thread> threads;
};

#endif
