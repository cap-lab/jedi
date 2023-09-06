
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tkdnn.h>

#include "int8_image_batch_stream.h"


ImageBatchStream::ImageBatchStream(nvinfer1::Dims dim, int batchSize, int maxBatches, const std::string& fileimglist, ImagePreprocessingOption preprocessingOption) {
    mBatchSize = batchSize;
    mMaxBatches = maxBatches;
    mDims = nvinfer1::Dims4{ dim.d[0], dim.d[1], dim.d[2], dim.d[3] };
    mHeight = dim.d[2];
    mWidth = dim.d[3];
    mImageSize = mDims.d[1]*mDims.d[2]*mDims.d[3];
    mBatch.resize(mBatchSize*mImageSize, 0);
    mFileBatch.resize(mDims.d[0]*mImageSize, 0);
    mFileImgList = fileimglist;
    readInListFile(fileimglist, mListImg);
	mPreprocessingOption = preprocessingOption;
	mInputBuffer = (float *) calloc(mImageSize * batchSize, sizeof(float));

    reset(0);
}

void ImageBatchStream::reset(int firstBatch) {
    mBatchCount = 0;
    mFileCount = 0;
    mFileBatchPos = mDims.d[0];
    skip(firstBatch);
}

// https://stackoverflow.com/questions/259297/how-do-you-copy-the-contents-of-an-array-to-a-stdvector-in-c-without-looping
// dataVec.insert(dataVec.end(), &dataArray[0], &dataArray[dataArraySize]);
bool ImageBatchStream::next() {
    std::cout<<"Next batch: "<<mBatchCount<<" of "<<mMaxBatches<<"\n";
    if (mBatchCount == mMaxBatches-1)
        return false;

    for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize) {
        assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
        if (mFileBatchPos == mDims.d[0] && !update())
            return false;

        csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
		memcpy(getBatch() + batchPos * mImageSize, getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize * sizeof(float));
		//getBatch().insert(getBatch().end(), , &(getFileBatch()[mFileBatchPos * mImageSize + csize * mImageSize]));
        //std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
    }
    mBatchCount++;
    return true;
}

void ImageBatchStream::skip(int skipCount) {
    if (mBatchSize >= mDims.d[0] && mBatchSize%mDims.d[0] == 0 && mFileBatchPos == mDims.d[0]) {
        mFileCount += skipCount * mBatchSize / mDims.d[0];
        return;
    }

    int x = mBatchCount;
    for (int i = 0; i < skipCount; i++)
        next();
    mBatchCount = x;
}

void ImageBatchStream::readInListFile(const std::string& dataFilePath, std::vector<std::string>& mListIn) {
    // dataFilePath contains the list of image paths
    int count = 0;
    FILE* f = fopen(dataFilePath.c_str(), "r");
    if (!f)
        FatalError("failed to open " + dataFilePath);
    
    char str[512];
    while (fgets(str, 512, f) != NULL) {
        for (int i = 0; str[i] != '\0'; ++i) {
            if (str[i] == '\n'){
                str[i] = '\0';
                break;
            }
        }
        count ++;
        mListIn.push_back(str);
        if(count == mMaxBatches)
            break;
    }
    fclose(f);
}

void ImageBatchStream::readCVimage(std::string inputFileName, float *input, bool fixshape) {
    // unaltered original DsImage
    cv::Mat m_OrigImage;
    // letterboxed DsImage given to the network as input
    cv::Mat m_LetterboxImage;
    m_OrigImage = cv::imread(inputFileName, cv::IMREAD_COLOR);

    if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
        FatalError("Unable to open " + inputFileName);

    int m_Height = m_OrigImage.rows;
    int m_Width = m_OrigImage.cols;
    if(fixshape) {
        m_Height = mHeight;
        m_Width = mWidth;
    }
	int original_width;
	int original_height;

	switch(mPreprocessingOption) {
		case LOAD_IMAGE_RESIZE:
			loadImageResize((char *)(inputFileName.c_str()), mWidth, mHeight, mDims.d[1], &original_width, &original_height, input);
			break;
		case LOAD_IMAGE_LETTERBOX:
			loadImageLetterBox((char *)(inputFileName.c_str()), mWidth, mHeight, mDims.d[1], &original_width, &original_height, input);
			break;
		case LOAD_IMAGE_RESIZE_NORM:
			loadImageResizeNorm((char *)inputFileName.c_str(), mWidth, mHeight, mDims.d[1], &original_width, &original_height, input);
			break;
		case LOAD_IMAGE_RESIZE_CROP_NORM:
			loadImageResizeCropNorm((char *)(inputFileName.c_str()), mWidth + 32, mHeight + 32, mDims.d[1], mWidth, input); // efficient former
			break;
		case LOAD_IMAGE_RESIZE_CROP:
			loadImageResizeCrop((char *) (inputFileName.c_str()), mWidth, mHeight, mDims.d[1], input); // efficient net
			break;
		case LOAD_IMAGE_RESIZE_CROP_NORM_ML:
			loadImageResizeCropNormML(inputFileName, mWidth, mHeight, mDims.d[1], input); // resnet mlperf
			break;
		default:
			break;
	}
}

bool ImageBatchStream::update() {
    std::string imgFileName = mListImg[mFileCount];
    mFileCount++;

    //read image
    mFileBatch.clear();
    readCVimage(imgFileName, mInputBuffer);
    
    mFileBatchPos = 0;
    return true;
}
