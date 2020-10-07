#include <stdio.h>
#include <stdlib.h>

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

static int get_coco_image_id(char *filename) {
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c)
        p = c;
    return atoi(p + 1);
}

std::map<int,std::list<std::string>> detected_map;
std::mutex mu;
int detected_num = 0;

void writeResultFile(std::string result_file_name) {
	int idx = 0, line_num = 0;
	std::ofstream result_file;

	result_file.open(result_file_name);
	result_file<<"["<<std::endl;
	while(!detected_map.empty()) {
		auto it = detected_map.find(idx);	
		if(it != detected_map.end()) {
			for(auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {	
				line_num++;
				result_file<<*it2;
				if(line_num != detected_num) {
					result_file<<", ";	
				}
				result_file<<std::endl;
			}

			detected_map.erase(it);
		}

		idx++;
	}
	result_file<<"]"<<std::endl;
	result_file.close();
}

static void print_coco_detections(Detection *dets, int nDets, int idx, int w, int h, char *path) {
	int i, j;
	int image_id = get_coco_image_id(path);
	std::list<std::string> detected;

	float x_ratio =  (float) w / (float) INPUT_WIDTH;
	float y_ratio =  (float) h / (float) INPUT_HEIGHT;

	for (i = 0; i < nDets; ++i) {
		if (dets[i].objectness >= 0) {
			float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
			float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
			float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
			float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

			xmin = x_ratio*xmin;
			xmax = x_ratio*xmax;
			ymin = y_ratio*ymin;
			ymax = y_ratio*ymax;

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
					detected.push_back(result.str());
					detected_num++;
				}
			}
		}
	}

	mu.lock();
	detected_map.insert(std::pair<int,std::list<std::string>>(idx,detected));
	mu.unlock();
}

void printDetector(Detection *dets, int idx, Dataset *dataset, int nDets) {
    int w = dataset->w[idx];
    int h = dataset->h[idx];
	char *path = (char *)(dataset->paths[idx].c_str());

	print_coco_detections(dets, nDets, idx, w, h, path);
}
