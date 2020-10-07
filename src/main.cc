#include <iostream>
#include <vector>

#include <tkDNN/tkdnn.h>

#include "config.h"
#include "model.h"
#include "dataset.h"
#include "thread.h"
#include "coco.h"

typedef struct _InstanceThreadData {
	PreProcessingThread *pre_thread;
	PostProcessingThread *post_thread;
	InferenceThread *infer_thread;
} InstanceThreadData;

void generateModels(int instance_num, ConfigData &config_data, std::vector<Model *> &models) {
	for(int iter = 0; iter < instance_num; iter++) {
		Model *model = new Model(&config_data, iter);

		model->initializeModel();
		model->initializeBuffers();

		models.emplace_back(model);	
	}
}

void generateDatasets(int instance_num, ConfigData &config_data, std::vector<Dataset *> &datasets) {
	for(int iter = 0; iter < instance_num; iter++) {
		Dataset *dataset = new Dataset(&config_data, iter);	

		dataset->initializeDataset();

		datasets.emplace_back(dataset);
	}
}

void runInstanceThread(void *d) {
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

void finalizeInstanceThreads(int instance_num, std::vector<PreProcessingThread *> &preProcessingThreads, std::vector<PostProcessingThread *> &postProcessingThreads, std::vector<InferenceThread *> &inferenceThreads) {
	for(int iter = 0; iter < instance_num; iter++) {
		delete preProcessingThreads[iter];		
		delete postProcessingThreads[iter];		
		delete inferenceThreads[iter];
	}
	
	preProcessingThreads.clear();
	postProcessingThreads.clear();
	inferenceThreads.clear();
}

void generateThreads(int instance_num, ConfigData &config_data, std::vector<Model *> models, std::vector<Dataset *> datasets) {
	std::vector<PreProcessingThread *> preProcessingThreads;
	std::vector<PostProcessingThread *> postProcessingThreads;
	std::vector<InferenceThread *> inferenceThreads;
	int signals[instance_num][MAX_DEVICE_NUM][MAX_BUFFER_NUM] = {0};
	std::vector<InstanceThreadData> instance_threads_data;
	std::vector<std::thread> instance_threads;
	
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

	for(int iter = 0; iter < instance_num; iter++) {
		instance_threads.push_back(std::thread(runInstanceThread, &(instance_threads_data[iter])));	
	}

	for(int iter = 0; iter < instance_num; iter++) {
		instance_threads[iter].join();	
	}

	writeResultFile();

	finalizeInstanceThreads(instance_num, preProcessingThreads, postProcessingThreads, inferenceThreads);
}

void finalizeData(int instance_num, std::vector<Model *> &models, std::vector<Dataset *> &datasets) {
	for(int iter = 0; iter < instance_num; iter++) {
		Model *model = models.at(iter);
		Dataset *dataset = datasets.at(iter);	

		model->finalizeBuffers();
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
	generateThreads(instance_num, config_data, models, datasets);

	// clear data
	finalizeData(instance_num, models, datasets);

	std::cout<<"End"<<std::endl;

	return 0;
}
