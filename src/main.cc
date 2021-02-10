#include <iostream>
#include <vector>

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

static void printHelpMessage() {
	std::cout<<"usage:"<<std::endl;
	std::cout<<"	./proc -c config_file [-r result_file] [-l power_log_file] [-t latency_log_file]" <<std::endl;
	std::cout<<"example:"<<std::endl;
	std::cout<<"	./proc -c yolov2.cfg"<<std::endl;
	std::cout<<"	./proc -c yolov2.cfg -r results/coco_results.json -l power.log -t latency.log"<<std::endl;
}

static void turnOnTegrastats(std::string power_file_name) {
	int result = -1;
	std::string cmd;

	cmd = std::string("rm -f ") + power_file_name;
	result = system(cmd.c_str());
	if(result == -1 || result == 127) {
		std::cerr<<"ERROR occurs at "<<__func__<<":"<<__LINE__<<std::endl;	
	}

	cmd = "tegrastats --start --logfile " + power_file_name + " --interval " + std::to_string(LOG_INTERVAL);
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

static void writeTimeResultFile(std::string time_file_name, double inference_time) {
	std::ofstream fp;

	fp.open(time_file_name.c_str());
	fp<<inference_time<<std::endl;
	fp.close();
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

static void generateThreads(int instance_num, ConfigData &config_data, std::string power_file_name, std::string time_file_name, std::vector<Model *> models, std::vector<Dataset *> datasets) {
	std::vector<PreProcessingThread *> preProcessingThreads;
	std::vector<PostProcessingThread *> postProcessingThreads;
	std::vector<InferenceThread *> inferenceThreads;
	int signals[instance_num][MAX_DEVICE_NUM+1][MAX_BUFFER_NUM] = {0};
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

	if(power_file_name.length() != 0) {
		turnOnTegrastats(std::string(power_file_name));
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

	if(power_file_name.length() != 0) {
		turnOffTegrastats();
	}

	if(time_file_name.length() != 0) {
		writeTimeResultFile(time_file_name, inference_time);
	}

	finalizeInstanceThreads(instance_num, preProcessingThreads, postProcessingThreads, inferenceThreads);
}

static void finalizeData(int instance_num, std::vector<Model *> &models, std::vector<Dataset *> &datasets) {
	for(int iter = 0; iter < instance_num; iter++) {
		Model *model = models.at(iter);
		Dataset *dataset = datasets.at(iter);	

		model->finalizeBuffers();
		model->finalizeModel();
		dataset->finalizeDataset();

		delete model;
		delete dataset;
	}

	models.clear();
	datasets.clear();
}

int main(int argc, char *argv[]) {
	int option;
	int instance_num = 0;
	std::string config_file_name = "config.cfg";
	std::string result_file_name = "coco_results.json";
	std::string power_file_name;
	std::string time_file_name;

	if(argc == 1) {
		printHelpMessage();
		return 0;
	}

	while((option = getopt(argc, argv, "c:r:p:t:h")) != -1) {
		switch(option) {
			case 'c':
				config_file_name = std::string(optarg);	
				break;
			case 'r':
				result_file_name = std::string(optarg);
				break;
			case 'p':
				power_file_name = std::string(optarg);
				break;
			case 't':
				time_file_name = std::string(optarg);
				break;
			case 'h':
				printHelpMessage();
				break;
		}	
	}

	// read configurations
	ConfigData config_data(config_file_name);
	instance_num = config_data.instance_num;

	// make models (engines, buffers)
	std::vector<Model *> models;
	generateModels(instance_num, config_data, models);

	// make dataset
	std::vector<Dataset *> datasets;
	generateDatasets(instance_num, config_data, datasets);

	// make threads
	generateThreads(instance_num, config_data, power_file_name, time_file_name, models, datasets);

	// write file
	writeResultFile(result_file_name);

	// clear data
	finalizeData(instance_num, models, datasets);

	return 0;
}
