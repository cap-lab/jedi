
#ifndef IMAGE_OPENCV_H
#define IMAGE_OPENCV_H

#include "image.h"


#ifdef __cplusplus
extern "C" {
#endif

void load_image_resize(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_OPENCV_H

