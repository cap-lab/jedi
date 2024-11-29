SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(TKDNN_APPLICATION_DIR "application/tkdnn")
set(ONNX_APPLICATION_DIR "application/onnx")
set(TKDNN_MODEL_DIR "model/tkdnn")
set(ONNX_MODEL_DIR "model/onnx")


set(DATASET_DIR "dataset")
set(RESULT_FORMAT_DIR "result_format")
set(KERNELS_DIR "kernels")

file(GLOB TKDNN_FILES ${TKDNN_APPLICATION_DIR}/*.cpp ${TKDNN_APPLICATION_DIR}/*.cc)
file(GLOB ONNX_FILES ${ONNX_APPLICATION_DIR}/*.cpp ${ONNX_APPLICATION_DIR}/*.cc)

file(GLOB TKDNN_MODEL_FILES ${TKDNN_MODEL_DIR}/*.cpp ${TKDNN_MODEL_DIR}/*.cc)
file(GLOB ONNX_MODEL_FILES ${ONNX_MODEL_DIR}/*.cpp ${ONNX_MODEL_DIR}/*.cc)

file(GLOB DATASET_FILES ${DATASET_DIR}/*.cpp ${DATASET_DIR}/*.cc)
file(GLOB RESULT_FORMAT_FILES ${RESULT_FORMAT_DIR}/*.cpp ${RESULT_FORMAT_DIR}/*.cc)

# message("TKDNN_FILES=${TKDNN_FILES}")
# message("ONNX_FILES=${ONNX_FILES}")
# message("DATASET_FILES=${DATASET_FILES}")
# message("DATASET_DIR=${DATASET_DIR}")

file(GLOB CUDA_KERNEL_FILES "${KERNELS_DIR}/*.cpp" "${KERNELS_DIR}/*.cu")

# message("CUDA_KERNEL_FILES=${CUDA_KERNEL_FILES}")

CUDA_ADD_LIBRARY(cuda_kernels ${CUDA_KERNEL_FILES})
target_include_directories(cuda_kernels INTERFACE ${KERNELS_DIR})

## tkdnn-dependent build (both tkdnn and onnx model are supported)
ADD_EXECUTABLE(proc main.cc config.cc model.cc stage.cc cuda_jedi.c binding.cc image_opencv.cc image.cc thread.cc box.cc detector.cc region_wrapper.cc yolo_wrapper.cc pillar.cc util.cc inference_application.cc ${TKDNN_MODEL_FILES} ${ONNX_MODEL_FILES} tensorrt_network.cc ${RESULT_FORMAT_FILES} ${DATASET_FILES} int8_image_batch_stream.cc int8_image_calibrator.cc int8_pillar_calibrator.cc ${TKDNN_FILES} ${ONNX_FILES})
target_include_directories(proc PRIVATE ${TKDNN_MODEL_DIR} ${ONNX_MODEL_DIR} ${TKDNN_APPLICATION_DIR} ${ONNX_APPLICATION_DIR} ${DATASET_DIR} ${RESULT_FORMAT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(proc m config++ nvinfer nvonnxparser nvinfer_plugin tkDNN stdc++ z ${CUDA_LIBRARIES} ${OpenCV_LIBS} cuda_kernels kernels)

## tkdnn-independent build (only onnx model is supported)
#ADD_EXECUTABLE(proc main.cc config.cc model.cc stage.cc cuda_jedi.c binding.cc image_opencv.cc image.cc thread.cc box.cc detector.cc region_wrapper.cc yolo_wrapper.cc pillar.cc util.cc inference_application.cc ${RESULT_FORMAT_FILES} ${DATASET_FILES} ${ONNX_MODEL_FILES} tensorrt_network.cc int8_image_batch_stream.cc int8_image_calibrator.cc int8_pillar_calibrator.cc ${ONNX_FILES})
#target_include_directories(proc PRIVATE ${ONNX_MODEL_DIR} ${ONNX_APPLICATION_DIR} ${DATASET_DIR} ${RESULT_FORMAT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
#TARGET_LINK_LIBRARIES(proc m config++ nvinfer nvonnxparser nvinfer_plugin stdc++ z ${CUDA_LIBRARIES} ${OpenCV_LIBS})

ADD_EXECUTABLE(network_analyzer analyzer.cc config.cc inference_application.cc ${TKDNN_FILES} ${DATASET_FILES} region_wrapper.cc yolo_wrapper.cc ${RESULT_FORMAT_FILES} util.cc image_opencv.cc box.cc cuda_jedi.c)
target_include_directories(network_analyzer PRIVATE ${TKDNN_MODEL_DIR} ${TKDNN_APPLICATION_DIR} ${DATASET_DIR} ${RESULT_FORMAT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(network_analyzer m config++ nvinfer tkDNN stdc++ ${CUDA_LIBRARIES} ${OpenCV_LIBS} cuda_kernels kernels)

ADD_EXECUTABLE(compute_coco_map coco_map.cc)
TARGET_LINK_LIBRARIES(compute_coco_map json-c jpeg nvinfer tkDNN kernels ${CUDA_LIBRARIES} ${OpenCV_LIBS})
