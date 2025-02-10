#ifndef IMAGE_OPENCV_H_
#define IMAGE_OPENCV_H_

#include "image.h"

#define IMAGE_COLOR_NUM (3)

extern float imagenet_mean[IMAGE_COLOR_NUM];
extern float imagenet_std[IMAGE_COLOR_NUM];

typedef enum _ResizeInterpolationOption {
	RESIZE_PILLOW_BILINEAR,
	RESIZE_PILLOW_BICUBIC,
	RESIZE_OPENCV_LINEAR,
	RESIZE_OPENCV_AREA,
} ResizeInterpolationOption;

void loadImageResize(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);
void loadImageLetterBox(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);

void loadImageResizeNorm(std::string filename, int w, int h, int c, int *orig_width, int *orig_height, float *input, float mean[IMAGE_COLOR_NUM]=imagenet_mean, float std[IMAGE_COLOR_NUM]=imagenet_std);
void loadImageResizeCropNorm(std::string filename, int w, int h, int c, int crop_size, ResizeInterpolationOption interpolation, float *input, float mean[IMAGE_COLOR_NUM]=imagenet_mean, float std[IMAGE_COLOR_NUM]=imagenet_std);

void loadImageResizeCrop(std::string filename, int w, int h, int c, float *input);

void loadImageLetterBoxNorm(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input, float mean[IMAGE_COLOR_NUM]=imagenet_mean, float std[IMAGE_COLOR_NUM]=imagenet_std);

void loadImageResizeCropNormML(std::string filename, int w, int h, int c, float *input);

typedef enum _ImagePreprocessingOption { 
	LOAD_IMAGE_RESIZE,
	LOAD_IMAGE_LETTERBOX,
	LOAD_IMAGE_RESIZE_NORM,
	LOAD_IMAGE_RESIZE_CROP_NORM,
	LOAD_IMAGE_RESIZE_CROP,
	LOAD_IMAGE_RESIZE_CROP_NORM_ML
} ImagePreprocessingOption;




#endif
