#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "common.h"

void error(const char *s) {
    perror(s);
    exit(-1);
}

void check_error(cudaError_t status) {
    // cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

float *cuda_make_array(float *x, size_t n) {
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
//    if (x) {
//        status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice, stream);
//        check_error(status);
//    } else {
//        check_error(cudaMemsetAsync(x_gpu, 0, size, stream));
//    }
	check_error(cudaMemset(x_gpu, 0, size));
    if (!x_gpu)
        error("Cuda malloc failed\n");
    return x_gpu;
}

float *cuda_make_array_host(size_t n) {
    float *x;
    size_t size = sizeof(float) * n;
	
	cudaError_t status = cudaHostAlloc((void **)&x, size, cudaHostAllocMapped);
    check_error(status);
    return x;
}

int *cuda_make_int_array_host(size_t n) {
    int *x;
    size_t size = sizeof(int) * n;

	cudaError_t status = cudaHostAlloc((void **)&x, size, cudaHostAllocMapped);
    check_error(status);
    return x;
}

int *cuda_make_int_array(int *x, size_t n) {
     int *x_gpu;
     size_t size = sizeof(int) * n;
     cudaError_t status = cudaMalloc((void **)&x_gpu, size);
     check_error(status);
//     if (x) {
//         status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
//         check_error(status);
//     }
	check_error(cudaMemset(x_gpu, 0, size));
     if (!x_gpu)
         error("Cuda malloc failed\n");
     return x_gpu;
}

void cuda_free(float *x_gpu) {
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

__half_raw *cuda_make_array_16(float *x, size_t n) {
    __half_raw *x_gpu;
    /* size_t size = sizeof(float) * n; */
    size_t size = 2 * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if (x) {
		// do nothing
    } else {
        check_error(cudaMemset(x_gpu, 0, size));
    }
    if (!x_gpu)
        error("cuda malloc failed\n");
    return x_gpu;
}

signed char *cuda_make_array_8(float *x, size_t n) {
    signed char *x_gpu;
    /* size_t size = sizeof(float) * n; */
    size_t size = n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if (x) {
		// do nothing
    } else {
        check_error(cudaMemset(x_gpu, 0, size));
    }
    if (!x_gpu)
        error("cuda malloc failed\n");
    return x_gpu;
}

