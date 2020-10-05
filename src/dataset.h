#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <vector>
#include <cassert>

#include "dataset.h"
#include "config.h"
#include "image.h"

class Dataset {
	public:
		ConfigData *config_data;
		int instance_id;

		std::vector<std::string> paths;
		std::vector<int> h;
		std::vector<int> w;
		int m;

		Dataset(ConfigData *config_data, int instance_id);
		void initializeDataset();
		void finalizeDataset();
};

#endif
