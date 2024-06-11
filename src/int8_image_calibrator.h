#ifndef INT8IMAGECALIBRATOR_H
#define INT8IMAGECALIBRATOR_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <stdint.h>
#include <iostream>
#include <string>
#include "NvInfer.h"

#include <fstream>
#include <iomanip>

#include "int8_image_batch_stream.h"

#include "utils.h"

/*
 * Int8ImageEntropyCalibrator implements the INT8 calibrator to achieve the
 * INT8 quantization. It uses a ImageBatchStream stream to scroll through 
 * images data. It also implements the calibration cache, a way to 
 * save the calibration process results to reduce the running time: 
 * the calibration process takes a long time.
 */
class Int8ImageEntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
	Int8ImageEntropyCalibrator(ImageBatchStream& stream, int firstBatch, const std::string& calibTableFilePath, 
							const std::string& inputBlobName, bool readCache = true);
	virtual ~Int8ImageEntropyCalibrator() { checkCuda(cudaFree(mDeviceInput)); }
	int getBatchSize() const NOEXCEPT override { return mStream.getBatchSize(); }
	bool getBatch(void* bindings[], const char* names[], int nbBindings) NOEXCEPT override;
	const void* readCalibrationCache(size_t& length) NOEXCEPT override;
	void writeCalibrationCache(const void* cache, size_t length) NOEXCEPT override;

private:
	ImageBatchStream mStream;
	const std::string mCalibTableFilePath{ nullptr };
	const std::string mInputBlobName;
	bool mReadCache{ true };

	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache;
};

#endif //INT8IMAGECALIBRATOR_H
