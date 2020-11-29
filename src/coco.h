#ifndef _COCO_H_
#define _COCO_H_

#include <iostream>

#include "box.h"
#include "config.h"
#include "variable.h"
#include "dataset.h"

void printDetector(Detection *dets, int idx, Dataset *dataset, int nDets);
void writeResultFile(std::string result_file_name);

#endif 
