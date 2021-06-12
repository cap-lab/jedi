#ifndef DATASET_H_
#define DATASET_H_

#include <iostream>
#include <vector>
#include <cassert>

/*
class Dataset {
	public:
		ConfigData *config_data;
		int instance_id;

		std::vector<std::string> paths;
		std::vector<int> h;
		std::vector<int> w;
		int m;

		Dataset(ConfigData *config_data, int instance_id);
		~Dataset();
		void initializeDataset();
		void finalizeDataset();
};
*/

template <typename T>
class Dataset {
	public:
		Dataset() {};
		virtual ~Dataset() {
			while(!data.empty()) {
				data.pop_back();
			}
		}
		virtual T *getData(int index) = 0;
		int getSize() {
			return data.size();
		}
	protected:
		std::vector<T> data;
};


#endif
