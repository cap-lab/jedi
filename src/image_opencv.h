#ifndef IMAGE_OPENCV_H_
#define IMAGE_OPENCV_H_

#include "image.h"

void loadImageResize(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);
void loadImageLetterBox(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);

void loadImageResizeNorm(std::string filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);
void loadImageResizeCropNorm(std::string filename, int w, int h, int c, int crop_size, float *input);

void loadImageResizeCrop(std::string filename, int w, int h, int c, float *input);

void loadImageLetterBoxNorm(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);

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
