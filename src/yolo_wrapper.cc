#include "yolo_wrapper.h"

int num_yolo_layer = 0;

struct yolo_size {
	int w;
	int h;
	int c;
};

struct yolo_size yolo_values[MAX_OUTPUT_NUM];

void setYoloValues(std::string network_name) {
	if(network_name == NETWORK_YOLOV4 || network_name == NETWORK_RESNEXT) {
		yolo_values[0].w = 52; yolo_values[0].h = 52; yolo_values[0].c = 255;
		yolo_values[1].w = 26; yolo_values[1].h = 26; yolo_values[1].c = 255;
		yolo_values[2].w = 13; yolo_values[2].h = 13; yolo_values[2].c = 255;
	}
	if(network_name == NETWORK_YOLOV4TINY) {
		yolo_values[0].w = 13; yolo_values[0].h = 13; yolo_values[0].c = 255;
		yolo_values[1].w = 26; yolo_values[1].h = 26; yolo_values[1].c = 255;
		yolo_values[2].w = 0; yolo_values[2].h = 0; yolo_values[2].c = 0;
	}
	else {
		yolo_values[0].w = 13; yolo_values[0].h = 13; yolo_values[0].c = 255;
		yolo_values[1].w = 26; yolo_values[1].h = 26; yolo_values[1].c = 255;
		yolo_values[2].w = 52; yolo_values[2].h = 52; yolo_values[2].c = 255;
	}
}

Box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	Box b;
	b.x = (i + x[index + 0*stride]) / lw;
	b.y = (j + x[index + 1*stride]) / lh;
	b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
	b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
	return b;
}

void correct_yolo_boxes(Detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
	int i;
	int new_w=0;
	int new_h=0;
	if (((float)netw/w) < ((float)neth/h)) {
		new_w = netw;
		new_h = (h * netw)/w;
	} else {
		new_h = neth;
		new_w = (w * neth)/h;
	}
	for (i = 0; i < n; ++i){
		Box b = dets[i].bbox;
		b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
		b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
		b.w *= (float)netw/new_w;
		b.h *= (float)neth/new_h;
		if(!relative){
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

float yolo_overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1/2;
	float l2 = x2 - w2/2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1/2;
	float r2 = x2 + w2/2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float yolo_box_intersection(Box a, Box b)
{
	float w = yolo_overlap(a.x, a.w, b.x, b.w);
	float h = yolo_overlap(a.y, a.h, b.y, b.h);
	if(w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float yolo_box_union(Box a, Box b)
{
	float i = yolo_box_intersection(a, b);
	float u = a.w*a.h + b.w*b.h - i;
	return u;
}

float yolo_box_iou(Box a, Box b)
{
	return yolo_box_intersection(a, b)/yolo_box_union(a, b);
}

int yolo_nms_comparator(const void *pa, const void *pb)
{
	Detection a = *(Detection *)pa;
	Detection b = *(Detection *)pb;
	float diff = 0;
	if(b.sort_class >= 0){
		diff = a.prob[b.sort_class] - b.prob[b.sort_class];
	} else {
		diff = a.objectness - b.objectness;
	}
	if(diff < 0) return 1;
	else if(diff > 0) return -1;
	return 0;
}

int entry_yolo_index(int b, int location, int entry, int width, int height, int channel) {
	int n =   location / (width*height);
	int loc = location % (width*height);

	return b*width*height*channel + n*width*height*(4+NUM_CLASSES+1) +
		entry*width*height + loc;
}


int yolo_computeDetections(float *predictions,  Detection *dets, int *ndets, int lw, int lh, int lc, float thresh, YoloData yolo) {

	int i,j,n;
	int count = *ndets;
	for (i = 0; i < lw*lh; ++i){
		int row = i / lw;
		int col = i % lw;
		for(n = 0; n < yolo.n_masks; ++n){
			int obj_index  = entry_yolo_index(0, n*lw*lh + i, 4, lw, lh, lc);
			float objectness = predictions[obj_index];
			if(objectness <= thresh) continue;
			int box_index  = entry_yolo_index(0, n*lw*lh + i, 0, lw, lh, lc);

			dets[count].bbox = get_yolo_box(predictions, yolo.bias, yolo.mask[n], box_index, col, row, lw, lh, INPUT_WIDTH, INPUT_HEIGHT, lw*lh);
			dets[count].objectness = objectness;
			dets[count].classes = NUM_CLASSES;
			for(j = 0; j < NUM_CLASSES; ++j){
				int class_index = entry_yolo_index(0, n*lw*lh + i, 4 + 1 + j, lw, lh, lc);
				float prob = objectness*predictions[class_index];
				dets[count].prob[j] = (prob > thresh) ? prob : 0;
			}

			++count;
			if(count >= MAX_DETECTION_BOXES)
				exit(-1);
		}
	}

	correct_yolo_boxes(dets + *ndets, count, INPUT_WIDTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_HEIGHT, 0);
	*ndets = count;
	return count;
}

void yolo_mergeDetections(Detection *dets, int ndets, int classes) {
	double nms_thresh = 0.45;
	int total = ndets;

	int i, j, k;
	k = total-1;
	for(i = 0; i <= k; ++i){
		if(dets[i].objectness == 0){
			Detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k+1;

	for(k = 0; k < classes; ++k){
		for(i = 0; i < total; ++i){
			dets[i].sort_class = k;
		}
		qsort(dets, total, sizeof(Detection), yolo_nms_comparator);
		for(i = 0; i < total; ++i){
			if(dets[i].prob[k] == 0) continue;
			Box a = dets[i].bbox;
			for(j = i+1; j < total; ++j){
				Box b = dets[j].bbox;
				if (yolo_box_iou(a, b) > nms_thresh){
					dets[j].prob[k] = 0;
				}
			}
		}
	}

}

void yoloLayerDetect(int batch, std::vector<float *> output_buffers, std::vector<YoloData> yolos, Detection *dets, std::vector<int> &detections_num) {
	int detection_num = 0;
	int output_num = output_buffers.size();
	int output_size = 0;

	for (int iter1 = 0; iter1 < batch; iter1++) {
		detection_num = 0;

		for(int iter2 = 0; iter2 < output_num; iter2++) {
			output_size = yolo_values[iter2].w * yolo_values[iter2].h * yolo_values[iter2].c;
			yolo_computeDetections(output_buffers[iter2] + output_size * iter1, &dets[iter1 * NBOXES], &detection_num, yolo_values[iter2].w, yolo_values[iter2].h, yolo_values[iter2].c, CONFIDENCE_THRESH, yolos[iter2]);
		}

		yolo_mergeDetections(&dets[iter1 * NBOXES], detection_num, NUM_CLASSES);
		detections_num[iter1] = detection_num;
	}
}
