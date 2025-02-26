
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "int8_image_batch_stream.h"


ImageBatchStream::ImageBatchStream(nvinfer1::Dims dim, int batchSize, int maxBatches, const std::string& fileimglist, ImagePreprocessingOption preprocessingOption) {
    if(batchSize > 0) {
        mBatchSize = batchSize;
    } else {
        mBatchSize = 1;
    }

    mMaxBatches = maxBatches;
    mDims = nvinfer1::Dims4{ dim.d[0], dim.d[1], dim.d[2], dim.d[3] };
    mHeight = dim.d[2];
    mWidth = dim.d[3];
    mImageSize = mDims.d[1]*mDims.d[2]*mDims.d[3];
    mBatch = reinterpret_cast<float*>(calloc(mImageSize*mBatchSize, sizeof(float)));
    mFileImgList = fileimglist;
    readInListFile(fileimglist, mListImg);
	mPreprocessingOption = preprocessingOption;
    mBatchCount = 0;
    mFileCount = 0;
}

ImageBatchStream::ImageBatchStream(nvinfer1::Dims dim, int batchSize, int maxBatches, const std::string& fileimglist,
                                ImagePreprocessingOption preprocessingOption, ResizeInterpolationOption interpolation, int crop_base_size, float *image_norm_mean, float *image_norm_std) :
                                ImageBatchStream(dim, batchSize, maxBatches, fileimglist, preprocessingOption) {
    mInterpolationOption = interpolation;
    mImageNormMean = image_norm_mean;
    mImageNormStd = image_norm_std;
    mCropBaseSize = crop_base_size;
}


// https://stackoverflow.com/questions/259297/how-do-you-copy-the-contents-of-an-array-to-a-stdvector-in-c-without-looping
// dataVec.insert(dataVec.end(), &dataArray[0], &dataArray[dataArraySize]);
bool ImageBatchStream::next() {
    std::cout<<"Next batch: "<<mBatchCount<<" of "<<mMaxBatches<<"\n";
    if (mBatchCount == mMaxBatches)
        return false;

    for (int batchPos = 0; batchPos < mBatchSize; batchPos += 1) {
        update(getBatch() + batchPos * mImageSize);
    }
    mBatchCount++;
    return true;
}

void ImageBatchStream::readInListFile(const std::string& dataFilePath, std::vector<std::string>& mListIn) {
    // dataFilePath contains the list of image paths
    int count = 0;
    FILE* f = fopen(dataFilePath.c_str(), "r");
    if (!f) {
        std::cerr << "[Error] Failed to open : " << dataFilePath << std::endl;
        exit(EXIT_FAILURE);
    }

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
        if(count == mMaxBatches * mBatchSize)
            break;
    }
    fclose(f);
}

void ImageBatchStream::readCVimage(std::string inputFileName, float *input, bool fixshape) {
    // unaltered original DsImage
    cv::Mat m_OrigImage;
    m_OrigImage = cv::imread(inputFileName, cv::IMREAD_COLOR);

    if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0) {
        std::cerr << "[Error] Unable to open : " << inputFileName << std::endl;
        exit(EXIT_FAILURE);
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
			loadImageResizeNorm((char *)inputFileName.c_str(), mWidth, mHeight, mDims.d[1], &original_width, &original_height, input, mImageNormMean, mImageNormStd);
			break;
		case LOAD_IMAGE_RESIZE_CROP_NORM:
			loadImageResizeCropNorm((char *)(inputFileName.c_str()), std::max(mCropBaseSize, mWidth), std::max(mCropBaseSize, mHeight), mDims.d[1], mWidth, mInterpolationOption, input, mImageNormMean, mImageNormStd); // efficient former
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

bool ImageBatchStream::update(float *inputBuffer) {
    std::string imgFileName = mListImg[mFileCount];
    mFileCount++;

    //read image
    //FileBatch.clear();
    readCVimage(imgFileName, inputBuffer);

    if(mListImg.size() <= mFileCount) {
        mFileCount = 0;
    }
    return true;
}
