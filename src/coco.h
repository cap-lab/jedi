#ifndef _COCO_H_
#define _COCO_H_

#include <iostream>

#include "box.h"
#include "config.h"
#include "variable.h"
// #include "dataset.h"

void writeResultFile(std::string result_file_name);
void detectCOCO(Detection *dets, int nDets, int idx, int w, int h, int iw, int ih, char *path);
void detectCOCO2(Detection *dets, int nDets, int w, int h, int iw, int ih);

#endif 
