#include "yolo_wrapper.h"

static int input_width, input_height;

Box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, int new_coords)
{
	Box b;
	if(new_coords == 0) 
	{
		b.x = (i + x[index + 0*stride]) / lw;
		b.y = (j + x[index + 1*stride]) / lh;
		b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
		b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
	}
	else
	{
		b.x = (i + x[index + 0 * stride]) / lw;
		b.y = (j + x[index + 1 * stride]) / lh;
		b.w = x[index + 2 * stride] * x[index + 2 * stride] * 4 * biases[2 * n] / w;
		b.h = x[index + 3 * stride] * x[index + 3 * stride] * 4 * biases[2 * n + 1] / h;
	}
	return b;
}

static void correct_yolo_boxes(Detection *dets, int n, int w, int h, int netw, int neth, int relative, bool letter_box)
{
	int i;
	int new_w=0;
	int new_h=0;
	if(letter_box == true)  {
		if (((float)netw/w) < ((float)neth/h)) {
			new_w = netw;
			new_h = (h * netw)/w;
		} else {
			new_h = neth;
			new_w = (w * neth)/h;
		}
	}
	else {
		new_w = netw;
    	new_h = neth;
	}

	float deltaw = netw - new_w;
	float deltah = neth - new_h;
	float ratiow = (float) new_w / netw;
	float ratioh = (float) new_h / neth;
	for (i = 0; i < n; ++i){
		Box b = dets[i].bbox;
        b.x = (b.x - deltaw / 2. / netw) / ratiow;
        b.y = (b.y - deltah / 2. / neth) / ratioh;
        b.w *= 1 / ratiow;
        b.h *= 1 / ratioh;
		if(!relative){
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

static float yolo_overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1/2;
	float l2 = x2 - w2/2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1/2;
	float r2 = x2 + w2/2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

static float yolo_box_intersection(Box a, Box b)
{
	float w = yolo_overlap(a.x, a.w, b.x, b.w);
	float h = yolo_overlap(a.y, a.h, b.y, b.h);
	if(w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

static float yolo_box_union(Box a, Box b)
{
	float i = yolo_box_intersection(a, b);
	float u = a.w*a.h + b.w*b.h - i;
	return u;
}

static float yolo_box_iou(Box a, Box b)
{
	return yolo_box_intersection(a, b)/yolo_box_union(a, b);
}

static int yolo_nms_comparator(const void *pa, const void *pb)
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

static int entry_yolo_index(int b, int location, int entry, int width, int height, int channel) {
	int n =   location / (width*height);
	int loc = location % (width*height);

	return b*width*height*channel + n*width*height*(4+NUM_CLASSES+1) +
		entry*width*height + loc;
}


static int yolo_computeDetections(float *predictions,  Detection *dets, int *ndets, int lw, int lh, int lc, float thresh, YoloData yolo, int orig_width, int orig_height, bool letter_box) {
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

			dets[count].bbox = get_yolo_box(predictions, yolo.bias, yolo.mask[n], box_index, col, row, lw, lh, input_width, input_height, lw*lh, yolo.new_coords);
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

	correct_yolo_boxes(dets + *ndets, count - *ndets, orig_width, orig_height, input_width, input_height, 0, letter_box);
	*ndets = count;
	return count;
}

static void box_c(const Box a, const Box b, float& top, float& bot, float& left, float& right) {
    top = std::min(a.y - a.h / 2, b.y - b.h / 2); 
    bot = std::max(a.y + a.h / 2, b.y + b.h / 2); 
    left = std::min(a.x - a.w / 2, b.x - b.w / 2); 
    right = std::max(a.x + a.w / 2, b.x + b.w / 2); 
}


// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
static float yolo_box_diou(const Box a, const Box b, const float nms_thresh=0.6)
{
    float top, bot, left, right;
    box_c(a, b, top, bot, left, right);
    float w = right - left;
    float h = bot - top;
    float c = w * w + h * h;
    float iou = yolo_box_iou(a, b); 
    if (c == 0)  
    	return iou;
    
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, nms_thresh);
    float diou_term = u;
    return iou - diou_term;
}


static void yolo_mergeDetections(Detection *dets, int ndets, int classes, double nms_thresh, tk::dnn::Yolo::nmsKind_t nms_kind) {
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
				if(nms_kind == tk::dnn::Yolo::GREEDY_NMS && yolo_box_iou(a, b) > nms_thresh) {
					dets[j].prob[k] = 0;
				}
				else if(nms_kind == tk::dnn::Yolo::DIOU_NMS && yolo_box_diou(a, b, nms_thresh) > nms_thresh) {
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

void yoloLayerDetect(Dataset *dataset, int sampleIndex, InputDim input_dim, bool letter_box, int batch, std::vector<float *> output_buffers, int buffer_id, std::vector<YoloData> yolos, Detection *dets, std::vector<int> &detections_num) {
	int detection_num = 0;
	int output_size = 0;
	int yolo_num = yolos.size();

	input_width = input_dim.width;
	input_height = input_dim.height;

	for (int iter1 = 0; iter1 < batch; iter1++) {
		int orig_width = dataset->w.at(sampleIndex * batch + iter1);
		int orig_height = dataset->h.at(sampleIndex * batch + iter1);
		detection_num = 0;

		for(int iter2 = 0; iter2 < yolo_num; iter2++) {
			int index = buffer_id * yolo_num + iter2;
			int w = yolos[iter2].width;
			int h = yolos[iter2].height;
			int c = yolos[iter2].channel;

			output_size = w * h * c;
			yolo_computeDetections(output_buffers[index] + output_size * iter1, &dets[iter1 * NBOXES], &detection_num, w, h, c, CONFIDENCE_THRESH, yolos[iter2], orig_width, orig_height, letter_box);
		}

		yolo_mergeDetections(&dets[iter1 * NBOXES], detection_num, NUM_CLASSES, yolos[0].nms_thresh, yolos[0].nms_kind);
		detections_num[iter1] = detection_num;
	}
}
