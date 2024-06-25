#ifndef INT8PILLARCALIBRATOR_H
#define INT8PILLARCALIBRATOR_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <stdint.h>
#include <iostream>
#include <string>
#include <map>
#include <NvInfer.h>

#include <fstream>
#include <iomanip>

#include "utils.h"
#include "lidar_dataset.h"

/*
 * Int8PillarEntropyCalibrator implements the INT8 calibrator to achieve the
 * INT8 quantization.
 */
//class Int8PillarEntropyCalibrator : public nvinfer1::IInt8MinMaxCalibrator  {
class Int8PillarEntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
	Int8PillarEntropyCalibrator(nvinfer1::INetworkDefinition *network, const std::string& fileLidarlist, const int calib_lidar_num, int batch,
                            const std::string calibTableFilePath, bool readCache = true);
	virtual ~Int8PillarEntropyCalibrator() { 
		for (auto iter = bindingMap.begin() ; iter !=  bindingMap.end(); iter++) {
			checkCuda(cudaFree(iter->second)); 
		}
		if(calibLidarSet != nullptr) {
			delete calibLidarSet;
		}
		if(fileInputBuf != nullptr) {
			free(fileInputBuf);
		}
		if(featureBuf != nullptr) {
			free(featureBuf);
		}
		if(indiceBuf != nullptr) {
			free(indiceBuf);
		}
	}
	int getBatchSize() const NOEXCEPT override { return mBatch; }
	bool getBatch(void* bindings[], const char* names[], int nbBindings) NOEXCEPT override;
	const void* readCalibrationCache(size_t& length) NOEXCEPT override;
	void writeCalibrationCache(const void* cache, size_t length) NOEXCEPT override;

private:
	const std::string mCalibTableFilePath{ nullptr };
	bool mReadCache{ true };

	std::vector<char> mCalibrationCache;
	std::map<std::string, void *> bindingMap;
	LidarDataset *calibLidarSet{ nullptr };
	float *fileInputBuf{ nullptr };
	float *featureBuf{ nullptr };
	int featureSize = 0;
	int *indiceBuf{ nullptr };
	int indiceSize = 0;
	int fileInputBufSize = 0;
	int mCalibLidarNum = 0;
	int calibLidarIndex = 0;
	int mBatch = 1;
};

#endif //INT8PILLARCALIBRATOR_H
