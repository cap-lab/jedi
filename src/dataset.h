#ifndef DATASET_H_
#define DATASET_H_

#include <iostream>
#include <vector>
#include <cassert>

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
