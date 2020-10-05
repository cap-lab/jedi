SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

ADD_EXECUTABLE(prototype main.cc config.cc model.cc cuda.c)
TARGET_LINK_LIBRARIES(prototype m config++ ${CUDNN_LIBRARY} nvinfer tkDNN stdc++ ${CUDA_LIBRARIES})
