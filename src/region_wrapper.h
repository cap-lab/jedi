#ifndef _REGION_WRAPPER_H_
#define _REGION_WRAPPER_H_

#include"config.h"
#include "variable.h"
#include "box.h"
#include "coco.h"

// #define IN_WIDTH    (INPUT_WIDTH / 32)
// #define IN_HEIGHT   (INPUT_HEIGHT / 32)
#define COORDS      4

void setBiases(std::string network_name);
void regionLayerDetect(InputDim input_dim, int batch, float *output, Detection *dets, int *detection_num);

#endif 
