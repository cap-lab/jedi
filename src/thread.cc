#include <iostream>
#include <thread>
#include <vector>

#include "thread.h"
#include "config.h"
#include "variable.h"
#include "model.h"
#include "detector.h"

Thread::Thread(ConfigData *config_data, int instance_id) {
	this->config_data = config_data;
	this->instance_id = instance_id;
	thread_num = 0;
}

Thread::~Thread() {
	threads.clear();
}

PreProcessingThread::~PreProcessingThread() {
	threads_data.clear();
}

PostProcessingThread::~PostProcessingThread() {
	threads_data.clear();
}

InferenceThread::~InferenceThread() {
	threads_data.clear();
}

void PreProcessingThread::setThreadData(std::vector<int> *signals, Model *model, Dataset *dataset, std::vector<long> *latency) {
	thread_num = config_data->instances.at(instance_id).pre_thread_num;	
	this->cur_running_index.assign(thread_num, 0);

	for(int iter = 0; iter < thread_num; iter++) {
		PreProcessingThreadData thread_data;		
		thread_data.config_data = config_data;
		thread_data.instance_id = instance_id;
		thread_data.tid = iter;
		thread_data.signals = signals;
		thread_data.dataset = dataset;
		thread_data.model = model;
		thread_data.latency = latency;
		thread_data.sample_index = &(this->sample_index);
		thread_data.mu = &(this->mu);
		thread_data.cur_running_index = &(this->cur_running_index);

		threads_data.push_back(thread_data);
	}
}

void PreProcessingThread::runThreads() {
	for(int iter = 0; iter < thread_num; iter++) {
		threads.push_back(std::thread(doPreProcessing, &(threads_data[iter])));
	}
}

void PreProcessingThread::joinThreads() {
	for(int iter = 0; iter < thread_num; iter++) {
		threads[iter].join();
	}
}

void PostProcessingThread::setThreadData(std::vector<int> *signals, Model *model, Dataset *dataset, std::vector<long> *latency) {
	thread_num = config_data->instances.at(instance_id).post_thread_num;	
	this->cur_running_index.assign(thread_num, 0);

	for(int iter = 0; iter < thread_num; iter++) {
		PostProcessingThreadData thread_data;		
		thread_data.config_data = config_data;
		thread_data.instance_id = instance_id;
		thread_data.tid = iter;
		thread_data.signals = signals;
		thread_data.model = model;
		thread_data.dataset = dataset;
		thread_data.latency = latency;
		thread_data.sample_index = &(this->sample_index);
		thread_data.mu = &(this->mu);
		thread_data.cur_running_index = &(this->cur_running_index);

		threads_data.push_back(thread_data);
	}
}

void PostProcessingThread::runThreads() {
	for(int iter = 0; iter < thread_num; iter++) {
		threads.push_back(std::thread(doPostProcessing, &(threads_data[iter])));
	}
}

void PostProcessingThread::joinThreads() {
	for(int iter = 0; iter < thread_num; iter++) {
		threads[iter].join();
	}
}

void InferenceThread::setThreadData(std::vector<int> *signals, Model *model) {
	thread_num = config_data->instances.at(instance_id).device_num;	

	for(int iter = 0; iter < thread_num; iter++) {
		InferenceThreadData thread_data;		
		thread_data.config_data = config_data;
		thread_data.instance_id = instance_id;
		thread_data.tid = iter;
		thread_data.curr_signals = &(signals[iter]);
		thread_data.next_signals = &(signals[iter+1]);
		thread_data.model = model;

		threads_data.push_back(thread_data);
	}
}

void InferenceThread::runThreads() {
	for(int iter = 0; iter < thread_num; iter++) {
		threads.push_back(std::thread(doInference, &(threads_data[iter])));
	}
}

void InferenceThread::joinThreads() {
	for(int iter = 0; iter < thread_num; iter++) {
		threads[iter].join();
	}
}
