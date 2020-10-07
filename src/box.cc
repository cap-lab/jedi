#include "box.h"

void allocateDetectionBox(int batch, Detection **dets) {
	*dets = (Detection *)calloc(batch * NBOXES, sizeof(Detection));
	for(int iter = 0; iter < batch * NBOXES; iter++) {
		(*dets)[iter].prob = (float *)calloc(NUM_CLASSES + 1, sizeof(float));	
	}
}

void deallocateDetectionBox(int n, Detection *dets) {
	int i;
	for (i = 0; i < n; ++i) {
		free(dets[i].prob);
		if (dets[i].mask)
			free(dets[i].mask);
	}
	free(dets);
}

static float overlap(float x1, float w1, float x2, float w2) {
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

static float box_intersection(Box a, Box b) {
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0)
		return 0;
	float area = w * h;
	return area;
}

static float box_union(Box a, Box b) {
	float i = box_intersection(a, b);
	float u = a.w * a.h + b.w * b.h - i;
	return u;
}

static float box_iou(Box a, Box b) { return box_intersection(a, b) / box_union(a, b); }

static int nms_comparator(const void *pa, const void *pb) {
	Detection a = *(Detection *)pa;
	Detection b = *(Detection *)pb;
	float diff = 0;
	if (b.sort_class >= 0) {
		diff = a.prob[b.sort_class] - b.prob[b.sort_class];
	} else {
		diff = a.objectness - b.objectness;
	}
	if (diff < 0)
		return 1;
	else if (diff > 0)
		return -1;
	return 0;
}

void do_nms_sort(Detection *dets, int total, float thresh) {
	int i, j, k;

	k = total - 1;
	for (i = 0; i <= k; ++i) {
		if (dets[i].objectness == 0) {
			Detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < NUM_CLASSES; ++k) {
		for (i = 0; i < total; ++i) {
			dets[i].sort_class = k;
		}
		qsort(dets, total, sizeof(Detection), nms_comparator);
		for (i = 0; i < total; ++i) {
			if (dets[i].prob[k] == 0)
				continue;
			Box a = dets[i].bbox;
			for (j = i + 1; j < total; ++j) {
				Box b = dets[j].bbox;
				if (box_iou(a, b) > thresh) {
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}
