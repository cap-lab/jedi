#ifndef _YOLO_WRAPPER_H_
#define _YOLO_WRAPPER_H_

#include <string.h>

#include"config.h"
#include "variable.h"
#include "model.h"
#include "coco.h"

void yoloLayerDetect(InputDim input_dim, int batch, std::vector<float *> output_buffers, int buffer_id, std::vector<YoloData> yolos, Detection *dets, std::vector<int> &detections_num);

#endif
