#include <iostream>
#include <vector>

//#include <tkDNN/tkdnn.h>

#include "config.h"
#include "variable.h"
#include "model.h"
#include "dataset.h"
#include "thread.h"
#include "coco.h"
#include "util.h"

#include "inference_application.h"

typedef struct _InstanceThreadData {
	PreProcessingThread *pre_thread;
	PostProcessingThread *post_thread;
	InferenceThread *infer_thread;
} InstanceThreadData;

bool exit_flag = false;

static void printHelpMessage() {
	std::cout<<"usage:"<<std::endl;
	std::cout<<"	./proc -c config_file [-r result_file] [-p power_log_file] [-t latency_log_file]" <<std::endl;
	std::cout<<"example:"<<std::endl;
	std::cout<<"	./proc -c yolov2.cfg"<<std::endl;
	std::cout<<"	./proc -c yolov2.cfg -r results/coco_results.json -p power.log -t latency.log"<<std::endl;
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

static void turnOnTegrastats(std::string power_file_name) {
	int result = -1;
	std::string cmd;

	turnOffTegrastats();

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

static void writeTimeResultFile(std::string time_file_name, double inference_time) {
	std::ofstream fp;

	fp.open(time_file_name.c_str());
	fp<<inference_time<<std::endl;
	fp.close();
}


static void generateModels(int instance_num, ConfigData &config_data, std::vector<Model *> &models, std::vector<IInferenceApplication *> &apps) {
	for(int iter = 0; iter < instance_num; iter++) {
		Model *model = nullptr;
		model = g_NetworkModelRegistry.create(config_data.instances.at(iter).model_type, &config_data, iter, apps[iter]);
		model->initializeModel();
		model->initializeBuffers();

		models.emplace_back(model);	
	}
}


static void initializePreAndPostprocessing(int instance_num, ConfigData &config_data, std::vector<IInferenceApplication *> &apps) {
	for(int iter = 0; iter < instance_num; iter++) {
		std::string network_name = config_data.instances.at(iter).network_name;
		int batch_size = config_data.instances.at(iter).batch;
		int pre_thread_num = config_data.instances.at(iter).pre_thread_num;
		int post_thread_num = config_data.instances.at(iter).post_thread_num;

		apps[iter]->initializePreprocessing(network_name, batch_size, pre_thread_num);
		apps[iter]->initializePostprocessing(network_name, batch_size, post_thread_num);
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

static void generateThreads(int instance_num, ConfigData &config_data, std::string power_file_name, std::string time_file_name,
							std::vector<Model *> models, std::vector<IInferenceApplication *> &apps) {
	std::vector<PreProcessingThread *> preProcessingThreads;
	std::vector<PostProcessingThread *> postProcessingThreads;
	std::vector<InferenceThread *> inferenceThreads;
	std::vector<InstanceThreadData> instance_threads_data;
	std::vector<long> latencies[instance_num];
	std::vector<std::thread> instance_threads;
	long start_time = 0;
	double inference_time = 0;
	
	for(int iter = 0; iter < instance_num; iter++) {
		int sample_size = config_data.instances.at(iter).sample_size;

		latencies[iter].assign(sample_size, 0);

		PreProcessingThread *pre_thread = new PreProcessingThread(&config_data, iter);
		pre_thread->setThreadData(models[iter], apps[iter], &(latencies[iter]));
		preProcessingThreads.push_back(pre_thread);

		PostProcessingThread *post_thread = new PostProcessingThread(&config_data, iter);
		post_thread->setThreadData(models[iter], apps[iter], &(latencies[iter]));
		postProcessingThreads.push_back(post_thread);

		InferenceThread *infer_thread = new InferenceThread(&config_data, iter);
		infer_thread->setThreadData(models[iter]);
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
	std::cout<< std::endl <<"inference time: "<<inference_time<<std::endl;

	if(power_file_name.length() != 0) {
		turnOffTegrastats();
	}

	if(time_file_name.length() != 0) {
		writeTimeResultFile(time_file_name, inference_time);
	}

	for(int iter = 0; iter < instance_num; iter++) {
		std::cout<< "average latency ("<< iter <<"): " << getAverageLatency(iter, &config_data, latencies[iter]) << std::endl;
	}

	finalizeInstanceThreads(instance_num, preProcessingThreads, postProcessingThreads, inferenceThreads);
}

static void finalizeData(int instance_num, std::vector<Model *> &models, std::vector<IInferenceApplication *> &apps) {
	for(int iter = 0; iter < instance_num; iter++) {
		Model *model = models.at(iter);
		IInferenceApplication *app = apps.at(iter);

		model->finalizeBuffers();
		model->finalizeModel();

		delete app;
		delete model;
	}

	models.clear();
	apps.clear();
}

int main(int argc, char *argv[]) {
	int option;
	int instance_num = 0;
	std::string config_file_name = "config.cfg";
	std::string result_file_name = "coco_results.json";
	std::string power_file_name;
	std::string time_file_name;
	cudaSetDeviceFlags(cudaDeviceMapHost);

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

	std::vector<IInferenceApplication *> apps;

	// read configurations
	ConfigData config_data(config_file_name, apps);
	instance_num = config_data.instance_num;

	// make models (engines, buffers)
	std::vector<Model *> models;
	generateModels(instance_num, config_data, models, apps);

	// initialize dataset, pre/post processing
	initializePreAndPostprocessing(instance_num, config_data, apps);

	// make threads
	generateThreads(instance_num, config_data, power_file_name, time_file_name, models, apps);

	// write file
	writeResultFile(result_file_name);

	// clear data
	finalizeData(instance_num, models, apps);

	if(exit_flag == false)
	{
		return 0;
	}
	else
	{
		return 1;
	}	
}
