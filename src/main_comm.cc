#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#include <tkDNN/tkdnn.h>

#include "config.h"
#include "variable.h"
#include "model.h"
#include "dataset.h"
#include "thread.h"
#include "coco.h"

std::string power_file_name;
std::string time_file_name;
int rcv_key_num, snd_key_num;
bool exit_flag;

// std::cerr<<__func__<<":"<<__LINE__<<std::endl;

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

static long getTime() {
	struct timespec time;
	if(0 != clock_gettime(CLOCK_REALTIME, &time)) {
		std::cerr<<"Something wrong on clock_gettime()"<<std::endl;		
		exit(-1);
	}
	return (time.tv_nsec) / 1000 + time.tv_sec * 1000000;
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

static int createMsgQueue(int key_num) {
	int msq_id;

	if(-1 == (msq_id = msgget((key_t)key_num, IPC_CREAT | 0666))) {
		perror( "msgget() failed");
		exit(1);
	}

	return msq_id;
}

static bool receiveDataFromMsgQueue(int rcv_msq_id, int type) {
	MsqData rcv_data;

	if(type == 1) {
		if(-1 == msgrcv(rcv_msq_id, &rcv_data, sizeof(MsqData) - sizeof(long), 1, 0)) {
			perror( "msgrcv() failed");
			exit(1);
		}
		return false;
	}
	else if(type == 2) {
		if(-1 != msgrcv(rcv_msq_id, &rcv_data, sizeof(MsqData) - sizeof(long), 2, IPC_NOWAIT)) {
			std::cerr<<__func__<<":"<<__LINE__<<" type two received"<<std::endl;
			return true;
		}
		return false;
	}
	
	return true;
}

static void sendDataToMsgQueue(int snd_msq_id) {
	MsqData snd_data;

	snd_data.type = 3;
	sprintf((char*)snd_data.buf, "inference done");	
	if ( -1 == msgsnd(snd_msq_id, &snd_data, sizeof(MsqData), 0)) {
		perror( "msgsnd() failed");
		exit(1);
	}
}

static bool checkMsg(int rcv_msq_id) {
	if(receiveDataFromMsgQueue(rcv_msq_id, 2)) {
		exit_flag = true;
		return true;
	}
	receiveDataFromMsgQueue(rcv_msq_id, 1);
	if(receiveDataFromMsgQueue(rcv_msq_id, 2)) {
		exit_flag = true;
		return true;
	}

	return false;
}

void generateThreads(int candidate, ConfigData config_data, std::vector<Model *> models, std::vector<Dataset *> datasets, std::string max_profile_file_name, std::string avg_profile_file_name) {
	std::vector<PreProcessingThread *> preProcessingThreads;
	std::vector<PostProcessingThread *> postProcessingThreads;
	std::vector<InferenceThread *> inferenceThreads;
	std::vector<long> inference_time_vec;
	std::vector<long> max_stage_time_vec;
	int signals[3] = {0};
	long start_time = 0;
	long max_stage_time = 0;
	int rcv_msq_id = -1, snd_msq_id = -1;

	exit_flag = false;
	rcv_msq_id = createMsgQueue(rcv_key_num);
	snd_msq_id = createMsgQueue(snd_key_num);

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

	// for(int titer1 = 0; titer1 < config_data.instances.at(candidate).sample_size; titer1++) {
	while(true) {
		// recv the msg from the controller
		if(checkMsg(rcv_msq_id)) {
			break;	
		}

		start_time = getTime();
		signals[0] = 0;
		while(!signals[0]) {
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

		signals[2] = 1;
		while(signals[2]) {
			usleep(SLEEP_TIME);	
		}
		inference_time_vec.push_back((long)(getTime() - start_time));

		// send the msg to the controller
		sendDataToMsgQueue(snd_msq_id);
	}

	signals[0] = 0;
	signals[2] = 1;
	pre_thread->joinThreads();
	post_thread->joinThreads();

	if(power_file_name.length() != 0) {
		turnOffTegrastats();
	}

	if(max_profile_file_name.length() != 0 && avg_profile_file_name.length() != 0) {
		models[candidate]->printProfile(max_profile_file_name, avg_profile_file_name);
	}

	if(time_file_name.length() != 0) {
		long max_latency = *std::max_element(inference_time_vec.begin(), inference_time_vec.end());
		long max_stage_time = *std::max_element(max_stage_time_vec.begin(), max_stage_time_vec.end());

		std::cerr<<"max_latency: "<<max_latency<<", max_stage_time: "<<max_stage_time<<std::endl;
		writeTimeResultFile(time_file_name, max_latency, max_stage_time);
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
	std::string result_file_name;
	std::string max_profile_file_name;
	std::string avg_profile_file_name;
	std::vector<ConfigData> config_data_vec;

	stickThisThreadToCore(6);

	if(argc == 1) {
		printHelpMessage();
		return 0;
	}

	while((option = getopt(argc, argv, "c:r:p:t:f:a:k:s:h")) != -1) {
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
			case 'l':
				config_list_file_name = std::string(optarg);
				break;
			case 'f':
				max_profile_file_name = std::string(optarg);
				break;
			case 'a':
				avg_profile_file_name = std::string(optarg);
				break;
			case 'k':
				rcv_key_num = atoi(optarg);
				break;
			case 's':
				snd_key_num = atoi(optarg);
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

	// make threads
	generateThreads(0, config_data, models, datasets, max_profile_file_name, avg_profile_file_name);
	// running_thread = std::thread(generateThreads, 0, config_data, models, max_profile_file_name, avg_profile_file_name);

	// write file
	if(result_file_name.length() != 0) {
		writeResultFile(result_file_name);
	}

	// clear data
	finalizeData(candidates_num, models, datasets);

	return 0;
}
