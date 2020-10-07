#include <iostream>
#include <vector>
#include <cstdlib>

#include <tkDNN/tkdnn.h>

#include "config.h"
#include "variable.h"
#include "model.h"
#include "dataset.h"
#include "thread.h"
#include "coco.h"

typedef struct _InstanceThreadData {
	PreProcessingThread *pre_thread;
	PostProcessingThread *post_thread;
	InferenceThread *infer_thread;
} InstanceThreadData;

static void turnOnTegrastats(std::string log_file_name) {
	int result = -1;
	std::string cmd;

	cmd = std::string("rm -f ") + log_file_name;
	result = system(cmd.c_str());
	if(result == -1 || result == 127) {
		std::cerr<<"ERROR occurs at "<<__func__<<":"<<__LINE__<<std::endl;	
	}

	cmd = "tegrastats --start --logfile " + log_file_name + " --interval " + std::to_string(LOG_INTERVAL);
	result = system(cmd.c_str());
	if(result == -1 || result == 127) {
		std::cerr<<"ERROR occurs at "<<__func__<<":"<<__LINE__<<std::endl;	
	}
}

static void turnOffTegrastats() {
	int result = -1;
	std::string cmd;
	
	cmd = "tegrastats --stop";
	result = system(cmd.c_str());
	if(result == -1 || result == 127) {
		std::cerr<<"ERROR occurs at "<<__func__<<":"<<__LINE__<<std::endl;	
	}
}

static long getTime() {
	struct timespec time;
	if(0 != clock_gettime(CLOCK_REALTIME, &time)) {
		std::cerr<<"Something wrong on clock_gettime()"<<std::endl;		
		exit(-1);
	}
	return (time.tv_nsec) / 1000 + time.tv_sec * 1000000;
}

static void generateModels(int instance_num, ConfigData &config_data, std::vector<Model *> &models) {
	for(int iter = 0; iter < instance_num; iter++) {
		Model *model = new Model(&config_data, iter);

		model->initializeModel();
		model->initializeBuffers();

		models.emplace_back(model);	
	}
}

static void generateDatasets(int instance_num, ConfigData &config_data, std::vector<Dataset *> &datasets) {
	for(int iter = 0; iter < instance_num; iter++) {
		Dataset *dataset = new Dataset(&config_data, iter);	

		dataset->initializeDataset();

		datasets.emplace_back(dataset);
	}
}

static void runInstanceThread(void *d) {
	InstanceThreadData *data = (InstanceThreadData *)d;
	PreProcessingThread *pre_thread = data-> pre_thread;	
	PostProcessingThread *post_thread = data-> post_thread;	
	InferenceThread *infer_thread = data->infer_thread;

	pre_thread->runThreads();
	post_thread->runThreads();
	infer_thread->runThreads();

	pre_thread->joinThreads();
	post_thread->joinThreads();
	infer_thread->joinThreads();
}

static void finalizeInstanceThreads(int instance_num, std::vector<PreProcessingThread *> &preProcessingThreads, std::vector<PostProcessingThread *> &postProcessingThreads, std::vector<InferenceThread *> &inferenceThreads) {
	for(int iter = 0; iter < instance_num; iter++) {
		delete preProcessingThreads[iter];		
		delete postProcessingThreads[iter];		
		delete inferenceThreads[iter];
	}
	
	preProcessingThreads.clear();
	postProcessingThreads.clear();
	inferenceThreads.clear();
}

static void generateThreads(int instance_num, ConfigData &config_data, char *log_file_name, std::vector<Model *> models, std::vector<Dataset *> datasets) {
	std::vector<PreProcessingThread *> preProcessingThreads;
	std::vector<PostProcessingThread *> postProcessingThreads;
	std::vector<InferenceThread *> inferenceThreads;
	int signals[instance_num][MAX_DEVICE_NUM][MAX_BUFFER_NUM] = {0};
	std::vector<InstanceThreadData> instance_threads_data;
	std::vector<std::thread> instance_threads;
	long start_time = 0;
	double inference_time = 0;
	
	for(int iter = 0; iter < instance_num; iter++) {
		int device_num = config_data.instances.at(iter).device_num;

		PreProcessingThread *pre_thread = new PreProcessingThread(&config_data, iter);
		pre_thread->setThreadData(signals[iter][0], models[iter], datasets[iter]);		
		preProcessingThreads.push_back(pre_thread);

		PostProcessingThread *post_thread = new PostProcessingThread(&config_data, iter);
		post_thread->setThreadData(signals[iter][device_num], models[iter], datasets[iter]);		
		postProcessingThreads.push_back(post_thread);

		InferenceThread *infer_thread = new InferenceThread(&config_data, iter);
		infer_thread->setThreadData(signals[iter][0], models[iter]);
		inferenceThreads.push_back(infer_thread);

		InstanceThreadData instance_thread_data;
		instance_thread_data.pre_thread = preProcessingThreads[iter];
		instance_thread_data.post_thread = postProcessingThreads[iter];
		instance_thread_data.infer_thread = inferenceThreads[iter];
		instance_threads_data.push_back(instance_thread_data);
	}

	if(log_file_name) {
		turnOnTegrastats(std::string(log_file_name));
	}	

	start_time = getTime();
	for(int iter = 0; iter < instance_num; iter++) {
		instance_threads.push_back(std::thread(runInstanceThread, &(instance_threads_data[iter])));	
	}

	for(int iter = 0; iter < instance_num; iter++) {
		instance_threads[iter].join();	
	}
	inference_time = (double)(getTime() - start_time) / 1000000;
	std::cout<<"inference time: "<<inference_time<<std::endl;

	if(log_file_name) {
		turnOffTegrastats();
	}	
	writeResultFile();

	finalizeInstanceThreads(instance_num, preProcessingThreads, postProcessingThreads, inferenceThreads);
}

static void finalizeData(int instance_num, std::vector<Model *> &models, std::vector<Dataset *> &datasets) {
	for(int iter = 0; iter < instance_num; iter++) {
		Model *model = models.at(iter);
		Dataset *dataset = datasets.at(iter);	

		model->finalizeBuffers();
		model->finalizeModel();
		dataset->finalizeDataset();
	}

	models.clear();
	datasets.clear();
}

int main(int argc, char *argv[]) {
	int instance_num = 0;

	std::cout<<"Start"<<std::endl;

	// read configurations
	ConfigData config_data(argv[1]);
	instance_num = config_data.instance_num;

	// make models (engines, buffers)
	std::vector<Model *> models;
	generateModels(instance_num, config_data, models);

	// make dataset
	std::vector<Dataset *> datasets;
	generateDatasets(instance_num, config_data, datasets);

	// make threads
	generateThreads(instance_num, config_data, argv[2], models, datasets);

	// clear data
	finalizeData(instance_num, models, datasets);

	std::cout<<"End"<<std::endl;

	return 0;
}
