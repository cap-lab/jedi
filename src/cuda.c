#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "common.h"

// const char *cublasGetErrorString(cublasStatus_t status) {
//     switch (status) {
//     case CUBLAS_STATUS_SUCCESS:
//         return "CUBLAS_STATUS_SUCCESS";
//     case CUBLAS_STATUS_NOT_INITIALIZED:
//         return "CUBLAS_STATUS_NOT_INITIALIZED";
//     case CUBLAS_STATUS_ALLOC_FAILED:
//         return "CUBLAS_STATUS_ALLOC_FAILED";
//     case CUBLAS_STATUS_INVALID_VALUE:
//         return "CUBLAS_STATUS_INVALID_VALUE";
//     case CUBLAS_STATUS_ARCH_MISMATCH:
//         return "CUBLAS_STATUS_ARCH_MISMATCH";
//     case CUBLAS_STATUS_MAPPING_ERROR:
//         return "CUBLAS_STATUS_MAPPING_ERROR";
//     case CUBLAS_STATUS_EXECUTION_FAILED:
//         return "CUBLAS_STATUS_EXECUTION_FAILED";
//     case CUBLAS_STATUS_INTERNAL_ERROR:
//         return "CUBLAS_STATUS_INTERNAL_ERROR";
//     case CUBLAS_STATUS_NOT_SUPPORTED:
//         return "CUBLAS_STATUS_NOT_SUPPORTED";
//     case CUBLAS_STATUS_LICENSE_ERROR:
//         return "CUBLAS_STATUS_LICENSE_ERROR";
//     }
//     return "unknown error";
// }

// void cuda_set_device(int n) {
//     cudaError_t status = cudaSetDevice(n);
//     check_error(status);
// }

// int cuda_get_device() {
//     int n = 0;
//     cudaError_t status = cudaGetDevice(&n);
//     check_error(status);
//     return n;
// }

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

// dim3 cuda_gridsize(size_t n) {
//     size_t k = (n - 1) / BLOCK + 1;
//     size_t x = k;
//     size_t y = 1;
//     if (x > 65536) {
//         x = ceil(sqrt(k));
//         y = (n - 1) / (x * BLOCK) + 1;
//     }
//     dim3 d = {x, y, 1};
//     return d;
// }

// cudnnHandle_t cudnn_handle() {
//     static int init[16] = {0};
//     static cudnnHandle_t handle[16];
//     int i = cuda_get_device();
//     if (!init[i]) {
//         cudnnCreate(&handle[i]);
//         init[i] = 1;
//     }
//     return handle[i];
// }

float *cuda_make_array(float *x, size_t n) {
    float *x_gpu;
    size_t size = sizeof(float) * n;
//    cudaStream_t stream;
//    cudnnGetStream(cudnn_handle(), &stream);

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

// int *cuda_make_int_array(int *x, size_t n) {
//     int *x_gpu;
//     size_t size = sizeof(int) * n;
//     cudaError_t status = cudaMalloc((void **)&x_gpu, size);
//     check_error(status);
//     if (x) {
//         status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
//         check_error(status);
//     }
//     if (!x_gpu)
//         error("Cuda malloc failed\n");
//     return x_gpu;
// }

void cuda_free(float *x_gpu) {
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

// void cuda_push_array(float *x_gpu, float *x, size_t n) {
//     size_t size = sizeof(float) * n;
//     cudaStream_t stream;
//     cudnnGetStream(cudnn_handle(), &stream);
//     cudaError_t status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice, stream);
//     check_error(status);
// }

// void cuda_pull_array(float *x_gpu, float *x, size_t n) {
//     size_t size = sizeof(float) * n;
//     cudaStream_t stream;
//     cudnnGetStream(cudnn_handle(), &stream);
//     cudaError_t status = cudaMemcpyAsync(x, x_gpu, size, cudaMemcpyDeviceToHost, stream);
//     check_error(status);
// }

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

