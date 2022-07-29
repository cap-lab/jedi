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

typedef struct _InputDim {
	int	width;
	int	height;
	int channel;
} InputDim;

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

#define MAX_DEVICE_NUM 12
#define DLA_NUM 2

#define NUM_CLASSES 80
#define PRINT_THRESH 0.3
#define CONFIDENCE_THRESH 0.3

// extern bool exit_flag;

#endif
