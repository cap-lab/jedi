cmake_minimum_required(VERSION 3.5)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

project(tinyJEDI VERSION 1.0.1 DESCRIPTION "tinyJEDI description")
set(DEFAULT_BUILD_TYPE "Release")

ADD_EXECUTABLE(proc main.cc config.cc model.cc cuda.c image_opencv.cc image.cc box.cc region_wrapper.cc yolo_wrapper.cc coco.cc inference_application.cc yolo_application.cc runner.cc)
TARGET_LINK_LIBRARIES(proc m config++ nvinfer tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBS} kernels)

include_directories(.)
add_library(tinyJEDI SHARED config.cc model.cc cuda.c image_opencv.cc image.cc box.cc region_wrapper.cc yolo_wrapper.cc coco.cc inference_application.cc yolo_application.cc runner.cc)
TARGET_LINK_LIBRARIES(tinyJEDI m config++ nvinfer tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBS} kernels)

install(TARGETS tinyJEDI DESTINATION /usr/lib)

install(FILES runner.h cuda.h coco.h model.h inference_application.h yolo_application.h region_wrapper.h yolo_wrapper.h image.h image_opencv.h variable.h config.h config_data.h box.h profiler.h DESTINATION include/tinyJEDI)
