#ifndef THREAD_H_
#define THREAD_H_ 

#include <iostream>
#include <thread>
#include <vector>

#include "thread.h"
#include "config.h"
#include "variable.h"
#include "dataset.h"
#include "model.h"
#include "detector.h"

typedef struct _PreProcessingThreadData {
	ConfigData *config_data;
	int instance_id;
	int tid;
	std::vector<int> *signals;
	Model *model;
	Dataset *dataset;
} PreProcessingThreadData;

typedef struct _PostProcessingThreadData {
	ConfigData *config_data;
	int instance_id;
	int tid;
	std::vector<int> *signals;
	Model *model;
	Dataset *dataset;
} PostProcessingThreadData;

typedef struct _InferenceThreadData {
	ConfigData *config_data;
	int instance_id;
	int tid;
	std::vector<int> *curr_signals;
	std::vector<int> *next_signals;
	Model *model;
} InferenceThreadData;

class Thread {
	public:
		ConfigData *config_data;
		int instance_id;
		int thread_num;
		std::vector<std::thread> threads;

		Thread(ConfigData *config_data, int instance_id);
		virtual ~Thread();
		virtual void runThreads() = 0;
		virtual void joinThreads() = 0;
};

class PreProcessingThread : public Thread {
	public:
		std::vector<PreProcessingThreadData> threads_data;

		PreProcessingThread(ConfigData *config_data, int instance_id) : Thread(config_data, instance_id) {};
		~PreProcessingThread();
		void setThreadData(std::vector<int> *signals, Model *model, Dataset *dataset);
		void runThreads();
		void joinThreads();
};

class PostProcessingThread : public Thread {
	public:
		std::vector<PostProcessingThreadData> threads_data;

		PostProcessingThread(ConfigData *config_data, int instance_id) : Thread(config_data, instance_id) {};
		~PostProcessingThread();
		void setThreadData(std::vector<int> *signals, Model *model, Dataset *dataset);
		void runThreads();
		void joinThreads();
};

class InferenceThread : public Thread {
	public:
		std::vector<InferenceThreadData> threads_data;

		InferenceThread(ConfigData *config_data, int instance_id) : Thread(config_data, instance_id) {};
		~InferenceThread();
		void setThreadData(std::vector<int> *signals, Model *model);
		void runThreads();
		void joinThreads();
};

#endif
