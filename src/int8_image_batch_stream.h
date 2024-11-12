#ifndef INT8IMAGEBATCHSTREAM_H
#define INT8IMAGEBATCHSTREAM_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <stdint.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <signal.h>
#include <stdlib.h>
#ifdef __linux__    
#include <unistd.h>
#endif

#include <mutex>

#include "NvInfer.h"
#include "image_opencv.h"


class ImageBatchStream {
public:
	ImageBatchStream(nvinfer1::Dims dim, int batchSize, int maxBatches, const std::string& fileimglist, ImagePreprocessingOption preprocessing_option);
	ImageBatchStream(nvinfer1::Dims dim, int batchSize, int maxBatches, const std::string& fileimglist, ImagePreprocessingOption preprocessingOption, float *image_norm_mean, float *image_norm_std);
	virtual ~ImageBatchStream() { free(mInputBuffer); }
	void reset(int firstBatch);
	bool next();
	void skip(int skipCount);
	float *getBatch() { return mBatch.data(); }
	int getBatchesRead() const { return mBatchCount; }
	int getBatchSize() const { return mBatchSize; }
	nvinfer1::Dims4 getDims() const { return mDims; }
	float* getFileBatch() { return mInputBuffer; }
	void readInListFile(const std::string& dataFilePath, std::vector<std::string>& mListIn);
	void readCVimage(std::string inputFileName, float *input, bool fixshape = true);
	bool update();

private:
	int mBatchSize{ 0 };
	int mMaxBatches{ 0 };
	int mBatchCount{ 0 };
	int mFileCount{ 0 };
	int mFileBatchPos{ 0 };
	int mImageSize{ 0 };

	nvinfer1::Dims4 mDims;
	std::vector<float> mBatch;
	std::vector<float> mFileBatch;
	float *mInputBuffer{nullptr};
	float *mImageNormMean{imagenet_mean};
	float *mImageNormStd{imagenet_std};

	int mHeight;
	int mWidth;
	std::string mFileImgList;
	std::vector<std::string> mListImg;
	ImagePreprocessingOption mPreprocessingOption{ LOAD_IMAGE_RESIZE };
}; 

#endif // INT8IMAGEBATCHSTREAM_H
