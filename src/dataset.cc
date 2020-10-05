#include <iostream>
#include <vector>
#include <cassert>

#include "dataset.h"
#include "config.h"
#include "image.h"

Dataset::Dataset(ConfigData *config_data, int instance_id) {
	this->config_data = config_data;
	this->instance_id = instance_id;
}

void Dataset::initializeDataset() {
	std::string image_path = config_data->instances.at(instance_id).image_path;

	getPaths((char *)image_path.c_str(), paths);
	m = paths.size();

	for(int iter = 0; iter < m; iter++) {
		h.emplace_back(0);	
		w.emplace_back(0);
	}
}

void Dataset::finalizeDataset() {
	while(!paths.empty()) {
		paths.pop_back();	
		h.pop_back();
		w.pop_back();
	}
}
