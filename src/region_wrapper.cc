#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <iostream>
#include <algorithm>

#include "cuda.h"
#include "box.h"
#include "region_wrapper.h"


static float pfBiases[2 * NUM_ANCHOR];

void setBiases(std::string network_name) {
	static float yolov2_biases[2 * NUM_ANCHOR] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
	static float densenet_biases[2 * NUM_ANCHOR] = {1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071};

	if(network_name == NETWORK_YOLOV2 || network_name == NETWORK_YOLOV2TINY) {
		for(int iter = 0; iter < 2 * NUM_ANCHOR; iter++) {
			pfBiases[iter] = yolov2_biases[iter];	
		}
	}
	else {
		for(int iter = 0; iter < 2 * NUM_ANCHOR; iter++) {
			pfBiases[iter] = densenet_biases[iter];	
		}
	}
}

static int entry_index(int batch, int location, int entry, int input_width, int input_height) {
    int w = input_width / 32;
    int h = input_height / 32;
    int n = location / (w * h);
    int outputs = h * w * NUM_ANCHOR * (NUM_CLASSES + COORDS + 1); 
    int loc = location % (w * h);

    return batch * outputs + n * w * h * (COORDS + NUM_CLASSES + 1) + entry * w * h + loc; 
}

static void correct_region_boxes(Detection *dets, int n, int w, int h, int input_width, int input_height) {
    int i;
    int new_w = 0;
    int new_h = 0;
    int netw = input_width;
    int neth = input_height;
	new_w = netw;
   	new_h = neth;

    for (i = 0; i < n; ++i){
        Box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw); 
        b.y =  (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth); 
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        
        b.x *= w;
        b.w *= w;
        b.y *= h;
        b.h *= h; 
        
        dets[i].bbox = b;
    }
}

static inline float logistic_activate(float x){
	return 1./(1. + exp(-x));
}

static Box get_region_box(float *x, int n, int index, int i, int j, int input_width, int input_height) {
	Box b;
    int w = input_width / 32;
    int h = input_height / 32;
    int stride = w * h;

	b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * pfBiases[2*n] / w;
    b.h = exp(x[index + 3*stride]) * pfBiases[2*n+1] / h;

    return b;
}

void get_region_detections(float *last_data, float thresh, InputDim input_dim, Detection *dets, int *nDets, int orig_width, int orig_height) {
	int w = input_dim.width;
	int h = input_dim.height;
    int lw = w / 32;
    int lh = h / 32;
    int i, j, n;
    float *predictions = last_data;

	for (i = 0; i < lw * lh; i++){
		int row = i / lw;
		int col = i % lw;
		for(n = 0; n < NUM_ANCHOR; ++n){
			int index = n * lw * lh + i;
			for(j = 0; j < NUM_CLASSES ; j++)
			{
				dets[index].prob[j] = 0;
			}
			int obj_index  = entry_index(0, index, COORDS, w, h);
			int box_index  = entry_index(0, index, 0, w, h);
			float scale = predictions[obj_index];
			dets[index].bbox = get_region_box(predictions, n, box_index, col, row, w, h);
			dets[index].objectness = scale > thresh ? scale : 0;
			float max = 0;
			for(j = 0; j < NUM_CLASSES; ++j){
				int c_index = entry_index(0, index, COORDS + 1 + j, w, h);
				float prob = scale * predictions[c_index];
				dets[index].prob[j] = (prob > thresh) ? prob : 0;
				if(prob > max) max = prob;
			}
			dets[index].prob[NUM_CLASSES] = max;
		}
	}
	correct_region_boxes(dets, lw*lh*NUM_ANCHOR, orig_width, orig_height, w, h);
	*nDets = lw *lh * NUM_ANCHOR ;
}

