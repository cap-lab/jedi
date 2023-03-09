#ifndef _COCO_H_
#define _COCO_H_

#include <iostream>
#include <string>
#include <list>

#include "box.h"
#include "config.h"
#include "variable.h"
#include "dataset.h"

void writeResultFile(std::string result_file_name);
void detectCOCO(Detection *dets, int nDets, int idx, int w, int h, int iw, int ih, char *path);
void addToDetectedMap(int image_index, std::list<std::string> detected);

#endif 
