#ifndef THREAD_H 
#define THREAD_H 

#include <iostream>
#include <thread>
#include <vector>

#include "thread.h"
#include "config.h"
#include "variable.h"

typedef struct _PreProcessingThreadData {
	ConfigData *config_data;
	int instance_id;
	int tid;
	int *signals;
} PreProcessingThreadData;

typedef struct _PostProcessingThreadData {
	ConfigData *config_data;
	int instance_id;
	int tid;
	int *signals;
} PostProcessingThreadData;

typedef struct _InferenceThreadData {
	ConfigData *config_data;
	int instance_id;
	int tid;
	int device_id;
	int *curr_signals;
	int *next_signals;
} InferenceThreadData;

class Thread {
	public:
		ConfigData *config_data;
		int instance_id;
		int thread_num;
		std::vector<std::thread> threads;

		Thread(ConfigData *config_data, int instance_id);
		virtual void setThreadData(int *signals) = 0;
		virtual void runThreads() = 0;
		virtual void joinThreads() = 0;
};

class PreProcessingThread : public Thread {
	public:
		std::vector<PreProcessingThreadData> threads_data;

		PreProcessingThread(ConfigData *config_data, int instance_id) : Thread(config_data, instance_id) {};
		void setThreadData(int *signals);
		void runThreads();
		void joinThreads();
};

class PostProcessingThread : public Thread {
	public:
		std::vector<PostProcessingThreadData> threads_data;

		PostProcessingThread(ConfigData *config_data, int instance_id) : Thread(config_data, instance_id) {};
		void setThreadData(int *signals);
		void runThreads();
		void joinThreads();
};

class InferenceThread : public Thread {
	public:
		std::vector<InferenceThreadData> threads_data;

		InferenceThread(ConfigData *config_data, int instance_id) : Thread(config_data, instance_id) {};
		void setThreadData(int *signals);
		void runThreads();
		void joinThreads();
};


#endif
