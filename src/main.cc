#include <iostream>
#include <vector>

#include <tkDNN/tkdnn.h>

#include "config.h"
#include "model.h"
#include "dataset.h"
#include "thread.h"

int main(int argc, char *argv[]) {
	int instance_num = 0, device_num = 0;

	std::cout<<"Start"<<std::endl;

	// read configurations
	ConfigData config_data(argv[1]);
	instance_num = config_data.instance_num;

	// make models (engines, buffers)
	std::vector<Model *> models;
	for(int iter = 0; iter < instance_num; iter++) {
		Model *model = new Model(&config_data, iter);

		model->initializeModel();
		model->initializeBuffers();

		models.emplace_back(model);	
	}

	// make dataset
	std::vector<Dataset *> datasets;
	for(int iter = 0; iter < instance_num; iter++) {
		Dataset *dataset = new Dataset(&config_data, iter);	

		dataset->initializeDataset();

		datasets.emplace_back(dataset);
	}

	// make threads
	std::vector<PreProcessingThread *> preProcessingThreads;
	std::vector<PostProcessingThread *> postProcessingThreads;
	std::vector<InferenceThread *> inferenceThreads;
	int signals[instance_num][MAX_DEVICE_NUM][MAX_BUFFER_NUM] = {0};
	
	for(int iter = 0; iter < instance_num; iter++) {
		device_num = config_data.instances.at(iter).device_num;

		PreProcessingThread *pre_thread = new PreProcessingThread(&config_data, iter);
		pre_thread->setThreadData(signals[iter][0]);		
		preProcessingThreads.push_back(pre_thread);

		PostProcessingThread *post_thread = new PostProcessingThread(&config_data, iter);
		post_thread->setThreadData(signals[iter][device_num]);		
		postProcessingThreads.push_back(post_thread);

		InferenceThread *infer_thread = new InferenceThread(&config_data, iter);
		infer_thread->setThreadData(signals[iter][0]);
		inferenceThreads.push_back(infer_thread);
	}

	std::cout<<"End"<<std::endl;

	return 0;
}
