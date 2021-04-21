SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

ADD_EXECUTABLE(proc main.cc config.cc model.cc cuda.c image_opencv.cc image.cc dataset.cc thread.cc box.cc detector.cc region_wrapper.cc yolo_wrapper.cc coco.cc util.cc)
TARGET_LINK_LIBRARIES(proc m config++ nvinfer tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

ADD_EXECUTABLE(network_analyzer analyzer.cc config.cc model.cc cuda.c image_opencv.cc image.cc dataset.cc thread.cc box.cc detector.cc region_wrapper.cc yolo_wrapper.cc coco.cc util.cc)
TARGET_LINK_LIBRARIES(network_analyzer m config++ nvinfer tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

ADD_EXECUTABLE(proc_comm main_comm.cc config.cc model.cc cuda.c image_opencv.cc image.cc dataset.cc thread.cc box.cc detector_comm.cc region_wrapper.cc yolo_wrapper.cc coco.cc util.cc)
TARGET_LINK_LIBRARIES(proc_comm config++ nvinfer tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})
