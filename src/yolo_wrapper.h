#ifndef _YOLO_WRAPPER_H_
#define _YOLO_WRAPPER_H_

#include <string.h>

#include "config.h"
#include "variable.h"
#include "model.h"
#include "coco.h"

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
} YoloData;

//void yoloLayerDetect(Dataset *dataset, int sampleIndex, InputDim input_dim, bool letter_box, int batch, std::vector<float *> output_buffers, int buffer_id, std::vector<YoloData> yolos, Detection *dets, std::vector<int> &detections_num);
int yolo_computeDetections(float *predictions,  Detection *dets, int *ndets, int lw, int lh, int lc, float thresh, YoloData yolo, int orig_width, int orig_height, int input_width, int input_height, bool letter_box);
void yolo_mergeDetections(Detection *dets, int ndets, double nms_thresh, tk::dnn::Yolo::nmsKind_t nms_kind);

#endif
