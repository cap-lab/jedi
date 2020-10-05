#include <iostream>
#include <vector>

#include <tkDNN/tkdnn.h>

#include "config.h"
#include "model.h"

int main(int argc, char *argv[]) {

	std::cout<<"Start"<<std::endl;

	// read configurations
	ConfigData config_data(argv[1]);

	// make models (engines, buffers)
	std::vector<Model *> models;
	for(int iter = 0; iter < config_data.instance_num; iter++) {
		Model *model = new Model(&config_data, iter);

		model->initializeModel();
		model->initializeBuffers();

		models.emplace_back(model);	
	}

	std::cout<<"End"<<std::endl;

	return 0;
}
