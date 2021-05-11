#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

// includes for OpenCV >= 3.x
#ifndef CV_VERSION_EPOCH
#include <opencv2/core/types.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#endif

// OpenCV includes for OpenCV 2.x
#ifdef CV_VERSION_EPOCH
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/version.hpp>
#endif

#include "variable.h"
#include "image.h"

typedef void* mat_cv;


static mat_cv *load_image_mat_cv(const char *filename, int flag)
{
	cv::Mat *mat_ptr = NULL;
	try {
		cv::Mat mat = cv::imread(filename, flag);
		if (mat.empty())
		{
			std::string shrinked_filename = filename;
			if (shrinked_filename.length() > 1024) {
				shrinked_filename.resize(1024);
				shrinked_filename = std::string("name is too long: ") + shrinked_filename;
			}
			std::cerr << "Cannot load image " << shrinked_filename << std::endl;

			return NULL;
		}
		cv::Mat dst;
		if (mat.channels() == 3) cv::cvtColor(mat, dst, cv::COLOR_RGB2BGR);
		else if (mat.channels() == 4) cv::cvtColor(mat, dst, cv::COLOR_RGBA2BGRA);
		else dst = mat;

		mat_ptr = new cv::Mat(dst);

		return (mat_cv *)mat_ptr;
	}
	catch (...) {
		std::cerr << "OpenCV exception: load_image_mat_cv" << std::endl;
	}
	if (mat_ptr) delete mat_ptr;
	return NULL;
}

static cv::Mat load_image_mat(char *filename, int channels)
{
	int flag = cv::IMREAD_UNCHANGED;
	if (channels == 0) flag = cv::IMREAD_COLOR;
	else if (channels == 1) flag = cv::IMREAD_GRAYSCALE;
	else if (channels == 3) flag = cv::IMREAD_COLOR;
	else {
		std::cerr << "OpenCV can't force load with " << channels << " channels"<< std::endl;
	}

	cv::Mat *mat_ptr = (cv::Mat *)load_image_mat_cv(filename, flag);

	if (mat_ptr == NULL) {
		return cv::Mat();
	}
	cv::Mat mat = *mat_ptr;
	delete mat_ptr;

	return mat;
}

static void mat_to_data(cv::Mat mat, float *input)
{
	int w = mat.cols;
	int h = mat.rows;
	int c = mat.channels();
	unsigned char *data = (unsigned char *)mat.data;
	int step = mat.step;
	for (int y = 0; y < h; ++y) {
		for (int k = 0; k < c; ++k) {
			for (int x = 0; x < w; ++x) {
				input[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
			}
		}
	}
}

cv::Mat letterbox(const cv::Mat& src, uchar pad) {
	int N = std::max(src.cols, src.rows);
	cv::Mat dst = cv::Mat::zeros(N, N, CV_8UC(src.channels()))
                + cv::Scalar(pad, pad, pad, 0);
	int dx = (N - src.cols) / 2;
	int dy = (N - src.rows) / 2;
	src.copyTo(dst(cv::Rect(dx, dy, src.cols, src.rows)));
	return dst;
}

#define GRAY_COLOR 127


void loadImageLetterBox(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input)
{
	try {
		cv::Mat loaded_image = load_image_mat(filename, c);

		*orig_width = loaded_image.cols;
		*orig_height = loaded_image.rows;

		int new_w = loaded_image.cols;
		int new_h = loaded_image.rows;
		if (((float)w / loaded_image.cols) < ((float)h / loaded_image.rows)) {
				new_w = w;
				new_h = (loaded_image.rows * w) / loaded_image.cols;
		}
		else {
				new_h = h;
				new_w = (loaded_image.cols * h) / loaded_image.rows;
		}

		cv::Mat resized(new_h, new_w, CV_8UC3);
		cv::resize(loaded_image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
		int N = std::max(new_w, new_h);
		cv::Mat dst = cv::Mat::zeros(w, h, CV_8UC3) + cv::Scalar(GRAY_COLOR, GRAY_COLOR, GRAY_COLOR, 0);
		int dx = (N - new_w) / 2;
		int dy = (N - new_h) / 2;
		resized.copyTo(dst(cv::Rect(dx, dy, new_w, new_h)));
		mat_to_data(dst, input);
	}
	catch (...) {
		std::cerr << " OpenCV exception: loadImageLetterBox() can't load image %s " << filename << std::endl;
	}
}

void loadImageResize(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input)
{
	try {
		cv::Mat loaded_image = load_image_mat(filename, c);

		*orig_width = loaded_image.cols;
		*orig_height = loaded_image.rows;

		cv::Mat resized(h, w, CV_8UC3);
		cv::resize(loaded_image, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
		mat_to_data(resized, input);
	}
	catch (...) {
		std::cerr << " OpenCV exception: loadImageResize() can't load image %s " << filename << std::endl;
	}
}
