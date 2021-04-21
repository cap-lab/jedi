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

void PreProcessingThread::setThreadData(std::vector<int> *signals, Model *model, Dataset *dataset) {
	thread_num = config_data->instances.at(instance_id).pre_thread_num;	

	for(int iter = 0; iter < thread_num; iter++) {
		PreProcessingThreadData thread_data;		
		thread_data.config_data = config_data;
		thread_data.instance_id = instance_id;
		thread_data.tid = iter;
		thread_data.signals = signals;
		thread_data.dataset = dataset;
		thread_data.model = model;

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

void PostProcessingThread::setThreadData(std::vector<int> *signals, Model *model, Dataset *dataset) {
	thread_num = config_data->instances.at(instance_id).post_thread_num;	

	for(int iter = 0; iter < thread_num; iter++) {
		PostProcessingThreadData thread_data;		
		thread_data.config_data = config_data;
		thread_data.instance_id = instance_id;
		thread_data.tid = iter;
		thread_data.signals = signals;
		thread_data.model = model;
		thread_data.dataset = dataset;

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

