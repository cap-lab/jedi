


#include "cnet_detections.h"


struct threshold : public thrust::binary_function<float,float,float>
{
  __host__ __device__
  float operator()(float x, float y) {
    double toll = 1e-6;
    if(fabsf(x-y)>toll)
        return 0.0f;
    else
        return x;
    }
};

void subtractWithThreshold_cnet(cudaStream_t stream, float *src_begin, float *src_end, float *src2_begin, float *src_out){
	struct threshold op;
    thrust::transform(thrust::cuda::par.on(stream), src_begin, src_end, src2_begin, src_out, op);
}

void sort_cnet(cudaStream_t stream, float *src_begin, float *src_end, int *idsrc){
    thrust::sort_by_key(thrust::cuda::par.on(stream), src_begin, src_end, idsrc, thrust::greater<float>());

}

void topk_cnet(cudaStream_t stream, float *src_begin, int *idsrc, int K, float *topk_scores,
            int *topk_inds){
    cudaMemcpyAsync(topk_scores, (float *)src_begin, K*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(topk_inds, idsrc, K*sizeof(int), cudaMemcpyDeviceToDevice, stream);
}

void topKxyclasses_cnet(cudaStream_t stream, int *ids_begin, int *ids_end, const int K, const int size, const int wh, int *clses, int *xs, int *ys){
    thrust::transform(thrust::cuda::par.on(stream), ids_begin, ids_end, thrust::make_constant_iterator(wh), clses, thrust::divides<int>());
    thrust::transform(thrust::cuda::par.on(stream), ids_begin, ids_end, thrust::make_constant_iterator(wh), ids_begin, thrust::modulus<int>());
    thrust::transform(thrust::cuda::par.on(stream), ids_begin, ids_end, thrust::make_constant_iterator(size), ys, thrust::divides<int>());
    thrust::transform(thrust::cuda::par.on(stream), ids_begin, ids_end, thrust::make_constant_iterator(size), xs, thrust::modulus<int>());
}

void topKxyAddOffset_cnet(cudaStream_t stream, int * ids_begin, const int K, const int size,
                     int *intxs_begin, int *intys_begin, float *xs_begin,
                     float *ys_begin, float *src_begin, float *src_out, int *ids_out){
    thrust::gather(thrust::cuda::par.on(stream), ids_begin, ids_begin + K, src_begin, src_out);
    thrust::transform(thrust::cuda::par.on(stream), intxs_begin, intxs_begin + K, src_out, xs_begin, thrust::plus<float>());
    thrust::transform(thrust::cuda::par.on(stream), ids_begin, ids_begin + K, thrust::make_constant_iterator(size), ids_out, thrust::plus<int>());
    thrust::gather(thrust::cuda::par.on(stream), ids_out, ids_out+K, src_begin, src_out);
    thrust::transform(thrust::cuda::par.on(stream), intys_begin, intys_begin + K, src_out, ys_begin, thrust::plus<float>());
}


void bboxes_cnet(cudaStream_t stream, int * ids_begin, const int K, const int size, float *xs_begin, float *ys_begin,
            float *src_begin, float *bbx0, float *bbx1, float *bby0, float *bby1,
            float *src_out, int *ids_out){
    thrust::gather(thrust::cuda::par.on(stream), ids_begin, ids_begin + K, src_begin, src_out);
    thrust::transform(thrust::cuda::par.on(stream), src_out, src_out + K, thrust::make_constant_iterator(2), src_out, thrust::divides<float>());
    // x0
    thrust::transform(thrust::cuda::par.on(stream), xs_begin, xs_begin + K, src_out, bbx0, thrust::minus<float>());
    // x1
    thrust::transform(thrust::cuda::par.on(stream), xs_begin, xs_begin + K, src_out, bbx1, thrust::plus<float>());
    thrust::transform(thrust::cuda::par.on(stream), ids_begin, ids_begin + K, thrust::make_constant_iterator(size), ids_out, thrust::plus<int>());
    thrust::gather(thrust::cuda::par.on(stream), ids_out, ids_out + K, src_begin, src_out);
    thrust::transform(thrust::cuda::par.on(stream), src_out, src_out + K, thrust::make_constant_iterator(2), src_out, thrust::divides<float>());
    // y0
    thrust::transform(thrust::cuda::par.on(stream), ys_begin, ys_begin + K, src_out, bby0, thrust::minus<float>());
    // y1
    thrust::transform(thrust::cuda::par.on(stream), ys_begin, ys_begin + K, src_out, bby1, thrust::plus<float>());

}


