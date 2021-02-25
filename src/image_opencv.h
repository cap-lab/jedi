#ifndef IMAGE_OPENCV_H_
#define IMAGE_OPENCV_H_

#include "image.h"

void loadImageResize(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);
void loadImage(char *filename, int input_size, float *input);

#endif
