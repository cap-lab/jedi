#ifndef CUDA_H_
#define CUDA_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
// #include <cudnn.h>

#define BLOCK 128

#if (CUDA_VERSION >= 9000)
typedef struct __half Half;
#else
typedef __half Half;
#define __hdiv hdiv
#endif

#ifdef __cplusplus
extern "C" {
#endif
// void cuda_push_array(float *x_gpu, float *x, size_t n);
// void cuda_pull_array(float *x_gpu, float *x, size_t n);
// void cuda_set_device(int n);
// const char *cublasGetErrorString(cublasStatus_t status);
void check_error(cudaError_t status);
float *cuda_make_array(float *x, size_t n);
float *cuda_make_array_host(size_t n);
// int *cuda_make_int_array(int *x, size_t n);
// float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
// dim3 cuda_gridsize(size_t n);
void cuda_free(float *x_gpu);

__half_raw *cuda_make_array_16(float *not_used, size_t n);
// void cuda_pull_array_16(__half_raw *x_gpu, float *x, size_t n);
// void cuda_push_array_16(__half_raw *x_gpu, float *x, size_t n);
// cudnnHandle_t cudnn_handle();

signed char *cuda_make_array_8(float *x, size_t n);


#ifdef __cplusplus
}
#endif

#endif
