#ifndef CNET_DETECTIONS_H_
#define CNET_DETECTIONS_H_

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/copy.h>

void sort_cnet(cudaStream_t stream, float *src_begin, float *src_end, int *idsrc);
void topk_cnet(cudaStream_t stream, float *src_begin, int *idsrc, int K, float *topk_scores, int *topk_inds);
void subtractWithThreshold_cnet(cudaStream_t stream, float *src_begin, float *src_end, float *src2_begin, float *src_out);
void topKxyclasses_cnet(cudaStream_t stream, int *ids_begin, int *ids_end, const int K, const int size, const int wh, int *clses, int *xs, int *ys);
void topKxyAddOffset_cnet(cudaStream_t stream, int * ids_begin, const int K, const int size,
                     int *intxs_begin, int *intys_begin, float *xs_begin,
                     float *ys_begin, float *src_begin, float *src_out, int *ids_out);
void bboxes_cnet(cudaStream_t stream, int * ids_begin, const int K, const int size, float *xs_begin, float *ys_begin,
            float *src_begin, float *bbx0, float *bbx1, float *bby0, float *bby1,
            float *src_out, int *ids_out);


#endif /* CNET_DETECTIONS_H_ */
