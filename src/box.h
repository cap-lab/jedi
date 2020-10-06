#ifndef _BOX_H_
#define _BOX_H_

typedef struct {
    float x, y, w, h;
} Box;

typedef struct Detection {
    Box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} Detection;

int entry_index(int batch, int location, int entry);
Detection *make_network_boxes(int *num);
void free_detections(Detection *dets, int n);

void do_nms_sort(Detection *dets, int total, float thresh);

#endif 
