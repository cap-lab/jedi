#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <iostream>
#include <thread>
#include <vector>

#include "model.h"
#include "config.h"
#include "variable.h"
#include "thread.h"
#include "image_opencv.h"
#include "region_wrapper.h"
#include "yolo_wrapper.h"
#include "coco.h"

void doPreProcessing(void *d);

void doPostProcessing(void *d);

void doInference(void *d);

#endif
