#ifndef _BOX_H_
#define _BOX_H_

#include <cstdlib>

#include "variable.h"

typedef struct {
    float x, y, w, h;
} Box;

typedef struct Detection {
    Box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} Detection;

#define MAX_DETECTION_BOXES (8192*2)
#define NBOXES MAX_DETECTION_BOXES

void allocateDetectionBox(int batch, Detection **dets);
void deallocateDetectionBox(int n, Detection *dets);
void do_nms_sort(Detection *dets, int total, float thresh);

#endif 
