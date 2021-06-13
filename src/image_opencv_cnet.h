
#ifndef IMAGE_OPENCV_CNET_H_
#define IMAGE_OPENCV_CNET_H_

#include <opencv2/opencv.hpp>

void loadImageAffineTransform(char *filename, int w, int h, int c, int *orig_width, int *orig_height, cv::Vec<float, 3> mean, cv::Vec<float, 3> stddev, cv::Mat dst, float *input);
cv::Mat restoreAffinedTransform(int orig_width, int orig_height, cv::Mat dst2);

#endif /* IMAGE_OPENCV_CNET_H_ */
