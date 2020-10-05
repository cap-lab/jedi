SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

ADD_EXECUTABLE(prototype main.cc config.cc model.cc cuda.c image_opencv.cc image.cc dataset.cc thread.cc)
TARGET_LINK_LIBRARIES(prototype m config++ ${CUDNN_LIBRARY} nvinfer tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})
