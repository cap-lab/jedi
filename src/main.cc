#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <tkDNN/tkdnn.h>

#include "config.h"
#include "variable.h"
#include "model.h"
#include "dataset.h"
#include "thread.h"
#include "coco.h"
#include "util.h"

std::string power_file_name;
std::string max_time_file_name;
std::string avg_time_file_name;

bool exit_flag = false;

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

static int stickThisThreadToCore(int core_id) {
	int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
	if (core_id < 0 || core_id >= num_cores)
		return EINVAL;

	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core_id, &cpuset);

	pthread_t current_thread = pthread_self();    
	return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

static void writeTimeResultFile(std::string time_file_name, double inference_time, long max_stage_time) {
	std::ofstream fp;

	fp.open(time_file_name.c_str());
	fp<<inference_time<<":"<<max_stage_time<<std::endl;
	fp.close();
}


static void generateModels(int candidates_num, ConfigData &config_data, std::vector<Model *> &models) {
	for(int iter = 0; iter < candidates_num; iter++) {
		Model *model = new Model(&config_data, iter);

		model->initializeModel();
		model->initializeBuffers();

		models.emplace_back(model);	
	}
}

static void generateDatasets(int candidates_num, ConfigData &config_data, std::vector<Dataset *> &datasets) {
	for(int iter = 0; iter < candidates_num; iter++) {
		Dataset *dataset = new Dataset(&config_data, iter);	

		dataset->initializeDataset();

		datasets.emplace_back(dataset);
	}
}

void generateThreads(int candidate, ConfigData config_data, std::vector<Model *> models, std::vector<Dataset *> datasets, std::string max_profile_file_name, std::string avg_profile_file_name) {
	std::vector<PreProcessingThread *> preProcessingThreads;
	std::vector<PostProcessingThread *> postProcessingThreads;
	std::vector<long> inference_time_vec;
	std::vector<long> max_stage_time_vec;
	std::vector<int> signals[MAX_DEVICE_NUM+1];
	long start_time = 0;
	long max_stage_time = 0;
	// double inference_time = 0;

	for(int iter = 0; iter < 3; iter++) {
		signals[iter].assign(1, 0);		
	}

	PreProcessingThread *pre_thread = new PreProcessingThread(&config_data, candidate);
	pre_thread->setThreadData(&(signals[0]), models[candidate], datasets[candidate]);		
	preProcessingThreads.push_back(pre_thread);

	PostProcessingThread *post_thread = new PostProcessingThread(&config_data, candidate);
	post_thread->setThreadData(&(signals[2]), models[candidate], datasets[candidate]);		
	postProcessingThreads.push_back(post_thread);

	if(power_file_name.length() != 0) {
		turnOnTegrastats(std::string(power_file_name));
	}

	pre_thread->runThreads();
	post_thread->runThreads();

	for(int titer1 = 0; titer1 < config_data.instances.at(candidate).sample_size; titer1++) {
		start_time = getTime();
		signals[0][0] = 0;
		while(!signals[0][0]) {
			usleep(SLEEP_TIME);	
		}

		max_stage_time = 0;
		for(int iter = 0; iter < config_data.instances.at(candidate).device_num; iter++) {
			long start_time2 = getTime();

			models[candidate]->infer(iter, 0);

			long stage_time  = getTime() - start_time2;
			if(stage_time > max_stage_time) {
				max_stage_time = stage_time;	
			}
		}
		max_stage_time_vec.push_back(max_stage_time);

		signals[2][0] = 1;
		while(signals[2][0]) {
			usleep(SLEEP_TIME);	
		}
		inference_time_vec.push_back((long)(getTime() - start_time));
	}
	pre_thread->joinThreads();
	post_thread->joinThreads();

	if(power_file_name.length() != 0) {
		turnOffTegrastats();
	}

	models[candidate]->printProfile(max_profile_file_name, avg_profile_file_name);

	if(max_time_file_name.length() != 0 && avg_time_file_name.length() != 0) {
		long max_latency = *std::max_element(inference_time_vec.begin(), inference_time_vec.end());
		long max_stage_time = *std::max_element(max_stage_time_vec.begin(), max_stage_time_vec.end());
		long avg_latency = std::accumulate(inference_time_vec.begin(), inference_time_vec.end(), 0) / inference_time_vec.size();
		long avg_stage_time = std::accumulate(max_stage_time_vec.begin(), max_stage_time_vec.end(), 0) / max_stage_time_vec.size();

		std::cerr<<"max_latency: "<<max_latency<<", max_stage_time: "<<max_stage_time<<std::endl;
		std::cerr<<"avg_latency: "<<avg_latency<<", avg_stage_time: "<<avg_stage_time<<std::endl;
		writeTimeResultFile(max_time_file_name, max_latency, max_stage_time);
		writeTimeResultFile(avg_time_file_name, avg_latency, avg_stage_time);
	}
}

static void finalizeData(int candidates_num, std::vector<Model *> &models, std::vector<Dataset *> &datasets) {
	for(int iter = 0; iter < candidates_num; iter++) {
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
	int option;
	int candidates_num;
	std::string config_list_file_name = "config_list.cfg";
	std::string config_file_name = "config.cfg";
	std::string result_file_name = "results/coco_results.json";
	std::string max_profile_file_name = "max_profile.log";
	std::string avg_profile_file_name = "avg_profile.log";
	std::vector<ConfigData> config_data_vec;

	stickThisThreadToCore(6);

	if(argc == 1) {
		printHelpMessage();
		return 0;
	}

	while((option = getopt(argc, argv, "c:r:p:t:y:f:a:h")) != -1) {
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
				max_time_file_name = std::string(optarg);
				break;
			case 'y':
				avg_time_file_name = std::string(optarg);
				break;
			case 'l':
				config_list_file_name = std::string(optarg);
				break;
			case 'f':
				max_profile_file_name = std::string(optarg);
				break;
			case 'a':
				avg_profile_file_name = std::string(optarg);
				break;
			case 'h':
				printHelpMessage();
				break;
		}	
	}

	// read configurations
	ConfigData config_data(config_file_name);
	candidates_num = config_data.instance_num;

	// make models (engines, buffers)
	std::vector<Model *> models;
	generateModels(candidates_num, config_data, models);

	// make dataset
	std::vector<Dataset *> datasets;
	generateDatasets(candidates_num, config_data, datasets);

	printf("wait");
	usleep(1000 * 1000 * 10);	
	printf("wait end");
	// make threads
	generateThreads(0, config_data, models, datasets, max_profile_file_name, avg_profile_file_name);

	// write file
	writeResultFile(result_file_name);

	// clear data
	finalizeData(candidates_num, models, datasets);

	if(exit_flag == false)
	{
		return 0;
	}
	else
	{
		return 1;
	}	
}
