cmake_minimum_required (VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/") 

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
find_package( CuDNN REQUIRED )

set (CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -g ")

if(${CUDNN_FOUND})
# Pass options to NVCC
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_72,code=sm_72)
include_directories (${CUDNN_INCLUDE_DIRS})
include_directories (/usr/local/cuda/include/)
endif(${CUDNN_FOUND})

add_subdirectory (src)
