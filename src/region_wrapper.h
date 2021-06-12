#ifndef _REGION_WRAPPER_H_
#define _REGION_WRAPPER_H_

#include"config.h"
#include "variable.h"
#include "box.h"
#include "coco.h"

#define COORDS     4
#define NUM_ANCHOR 5


void setBiases(std::string network_name);
void get_region_detections(float *last_data, float thresh, InputDim input_dim, Detection *dets, int *nDets, int orig_width, int orig_height);

#endif 
