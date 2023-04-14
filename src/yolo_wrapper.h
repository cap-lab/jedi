#ifndef _YOLO_WRAPPER_H_
#define _YOLO_WRAPPER_H_

#include <string.h>

#include <tkDNN/tkdnn.h>

#include "config.h"
#include "variable.h"
#include "coco_format.h"

typedef struct _YoloData {
	float *mask;
	int n_masks;
	float *bias;
	int new_coords;
	double nms_thresh;
	tk::dnn::Yolo::nmsKind_t nms_kind;
	int height;
	int width;
	int channel;
	float scale_x_y;
	int num;
} YoloData;

int entry_yolo_index(int b, int location, int entry, int width, int height, int channel);
//void yoloLayerDetect(Dataset *dataset, int sampleIndex, InputDim input_dim, bool letter_box, int batch, std::vector<float *> output_buffers, int buffer_id, std::vector<YoloData> yolos, Detection *dets, std::vector<int> &detections_num);
int yolo_computeDetections(float *predictions,  Detection *dets, int *ndets, int lw, int lh, int lc, float thresh, YoloData yolo, int orig_width, int orig_height, int input_width, int input_height, bool letter_box);
void yolo_mergeDetections(Detection *dets, int ndets, double nms_thresh, tk::dnn::Yolo::nmsKind_t nms_kind);

#endif
