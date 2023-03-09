#ifndef IMAGE_OPENCV_H_
#define IMAGE_OPENCV_H_

#include "image.h"

void loadImageResize(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);
void loadImageLetterBox(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);

void loadImageResizeNorm(std::string filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);


#endif
