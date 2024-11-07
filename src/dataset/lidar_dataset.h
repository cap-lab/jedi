#ifndef LIDAR_DATASET_H_
#define LIDAR_DATASET_H_

#include <iostream>
#include <vector>
#include <cassert>

#include "dataset.h"

typedef struct _LidarData {
  std::string path;
} LidarData;

class LidarDataset : public Dataset<LidarData> {
	public:
		LidarDataset(std::string lidarBinListFile);
		~LidarDataset();
		virtual LidarData *getData(int index) override;
	private:
		std::string lidarBinListFile;
		void fillLidarPath(char *filename);
};

#endif
