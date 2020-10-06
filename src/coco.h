#ifndef _COCO_H_
#define _COCO_H_

#include "box.h"
#include "dataset.h"
#include "variable.h"

void printDetector(Detection *dets, int idx, Dataset *dataset, int nDets);
void writeResultFile();

#endif 
