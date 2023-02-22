SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

CUDA_ADD_LIBRARY(cuda_kernels cnet_detections.cu)

ADD_EXECUTABLE(proc main.cc config.cc model.cc stage.cc cuda.c image_opencv.cc image.cc thread.cc box.cc detector.cc region_wrapper.cc yolo_wrapper.cc coco.cc util.cc inference_application.cc yolo_application.cc image_dataset.cc cnet_application.cc tkdnn_model.cc onnx_model.cc yolo_onnx_application.cc tensorrt_network.cc)
TARGET_LINK_LIBRARIES(proc m config++ nvinfer nvonnxparser tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBS} cuda_kernels kernels)

ADD_EXECUTABLE(network_analyzer analyzer.cc config.cc inference_application.cc yolo_application.cc image_dataset.cc region_wrapper.cc yolo_wrapper.cc coco.cc util.cc image_opencv.cc box.cc cnet_application.cc cuda.c)
TARGET_LINK_LIBRARIES(network_analyzer m config++ nvinfer tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBS} cuda_kernels kernels)

ADD_EXECUTABLE(compute_coco_map coco_map.cc)
TARGET_LINK_LIBRARIES(compute_coco_map json-c jpeg nvinfer tkDNN kernels ${CUDA_LIBRARIES} ${OpenCV_LIBS})
