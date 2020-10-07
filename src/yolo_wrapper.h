#ifndef _YOLO_WRAPPER_H_
#define _YOLO_WRAPPER_H_

#include <string.h>

#include"config.h"
#include "variable.h"
#include "model.h"
#include "coco.h"

void yoloLayerDetect(int batch, std::vector<float *> output_buffers, int buffer_id, int output_num, std::vector<YoloData> yolos, std::vector<YoloValue> yolo_values, Detection *dets, std::vector<int> &detections_num);

#endif
