#ifndef VARIABLE_H_
#define VARIABLE_H_

enum DeviceType {
	DEVICE_GPU,
	DEVICE_DLA,
};

enum DataType {
	TYPE_FP32,
	TYPE_FP16,
	TYPE_INT8,
};

#define NETWORK_YOLOV2 "yolo2"
#define NETWORK_YOLOV2TINY "yolo2tiny"
#define NETWORK_YOLOV3 "yolo3"
#define NETWORK_YOLOV3TINY "yolo3tiny"
#define NETWORK_YOLOV4 "yolo4"
#define NETWORK_YOLOV4TINY "yolo4tiny"
#define NETWORK_RESNEXT "cspresnext"
#define NETWORK_DENSENET "densenet"

#define STRING_SIZE 100
#define SLEEP_TIME 100
#define LOG_INTERVAL 100

#define MAX_DEVICE_NUM 4
#define MAX_BUFFER_NUM 8
#define MAX_OUTPUT_NUM 4

#define NUM_CLASSES 80
#define INPUT_WIDTH 608
#define INPUT_HEIGHT 608
#define INPUT_CHANNEL 3
#define INPUT_SIZE (INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNEL)
#define REGION_WIDTH (INPUT_WIDTH / 32)
#define REGION_HEIGHT (INPUT_HEIGHT / 32)

#define NMS 0.45
#define PRINT_THRESH 0.3
#define CONFIDENCE_THRESH 0.005
#define NUM_ANCHOR 5
#define MAX_DETECTION_BOXES 8192
#define NBOXES MAX_DETECTION_BOXES

#endif
