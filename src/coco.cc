#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "variable.h"
#include "config.h"
#include "coco.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <list>
#include <vector>
#include <mutex>

static int coco_ids[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
						11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
						22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
						35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
						46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
						56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
						67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    					80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

const char *coco_class_name[] = { 
	"person", "bicycle", "car", "motorcycle", "airplane", 
	"bus", "train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
	"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"
}; 

std::vector<std::string> classesNames = std::vector<std::string>(coco_class_name, std::end( coco_class_name));

static int get_coco_image_id(char *filename) {
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c)
        p = c;
    return atoi(p + 1);
}

static std::map<int,std::list<std::string>> detected_map;
static std::mutex mu;
static int detected_num = 0;
static int image_id = 0; 

void writeResultFile(std::string result_file_name) {
	int idx = 0, line_num = 0;
	std::ofstream result_file;
	std::vector<std::string> results_vec;

	while(!detected_map.empty() && line_num < detected_num) {
		auto it = detected_map.find(idx);	
		if(it != detected_map.end()) {
			for(auto it2 = it->second.begin(); it2 != it->second.end() && line_num < detected_num; it2++) {	
				line_num++;
				results_vec.push_back(*it2);
			}

			detected_map.erase(it);
		}

		idx++;
	}

	result_file.open(result_file_name);
	result_file<<"["<<std::endl;
	if(results_vec.size() > 0) {
		auto it = results_vec.begin();
		for(; it != std::prev(results_vec.end()); it++) {
			result_file<<*it;
			result_file<<","<<std::endl;
		}
		result_file<<*it<<std::endl;
	}
	result_file<<"]";
	result_file.close();
}

void detectCOCO2(Detection *dets, int nDets, int w, int h, int iw, int ih) {
	int i, j;
	std::list<std::string> detected;

	for (i = 0; i < nDets; ++i) {
		if (dets[i].objectness >= 0) {
			float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
			float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
			float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
			float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

			if (xmin < 0)
				xmin = 0;
			if (ymin < 0)
				ymin = 0;
			if (xmax > w)
				xmax = w;
			if (ymax > h)
				ymax = h;

			float bx = xmin;
			float by = ymin;
			float bw = xmax - xmin;
			float bh = ymax - ymin;

			for (j = 0; j < NUM_CLASSES; ++j) {
				if (dets[i].prob[j] >= PRINT_THRESH) {
					std::stringstream result;

					result<<"{\"image_id\":"<<image_id<<", \"category_id\":"<<coco_ids[j]<<", \"bbox\":["<<bx<<", "<<by<<", "<<bw<<", "<<bh<<"], \"score\":"<<dets[i].prob[j]<<"}";
					// std::cout<<"{\"image_id\":"<<image_id<<", \"category_id\":"<<coco_ids[j]<<", \"bbox\":["<<bx<<", "<<by<<", "<<bw<<", "<<bh<<"], \"score\":"<<dets[i].prob[j]<<"}"<<std::endl;

					detected.push_back(result.str());
					detected_num++;
				}
			}
		}
	}

	detected_map.insert(std::pair<int,std::list<std::string>>(image_id, detected));
	image_id++;
}

void drawBox(int width, int height, int step, char *input_data, Detection *dets, std::vector<int> detections_num, char *result_file_name)
{
	cv::Mat frame = cv::Mat(height, width, CV_8UC3, input_data, step);
	int x0, x1, y0, y1;
	int baseline = 0;
	float font_scale = 0.5;
	int thickness = 1;

	// cv::imwrite("input.jpg", frame);

	// assuming batch is zero
	for(int i=0; i<detections_num[0]; ++i) { 
		if(dets[i].objectness >= 0) {
			float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
			float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
			float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
			float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

			if (xmin < 0)
				xmin = 0;
			if (ymin < 0)
				ymin = 0;
			if (xmax > width)
				xmax = width;
			if (ymax > height)
				ymax = height;

			float bx = xmin;
			float by = ymin;
			float bw = xmax - xmin;
			float bh = ymax - ymin;

			x0 = bx;
			x1 = bx + bw;
			y0 = by;
			y1 = by + bh;

			for (int j = 0; j < NUM_CLASSES; ++j) {
				if (dets[i].prob[j] >= PRINT_THRESH) {
					// draw rectangle
					cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 0, 255), thickness); 

					std::string det_class = classesNames[j];
					cv::Size text_size = cv::getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
					cv::rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), cv::Scalar(0, 0, 255), -1); 
					cv::putText(frame, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
				}
			}
		}
	}

	cv::imwrite(result_file_name, frame);
}
