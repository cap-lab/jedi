#include "binding.h"

int getDataTypeSize(nvinfer1::DataType data_type) {
    int data_type_size = 0;

    switch(data_type) {
#if NV_TENSORRT_MAJOR > 8
        case nvinfer1::DataType::kINT64:
            data_type_size = sizeof(int64_t);
            break;
#endif
        case nvinfer1::DataType::kFLOAT:
            data_type_size = sizeof(float);
            break;
        // 4 bytes are allocated for other data types
        default:
        // case nvinfer1::DataType::kHALF:
        // case nvinfer1::DataType::kINT8:
        // case nvinfer1::DataType::kINT32:
        // case nvinfer1::DataType::kBOOL:
        // case nvinfer1::DataType::kUINT8:
        // case nvinfer1::DataType::kFP8:
        // case nvinfer1::DataType::kBF16:	
        // case nvinfer1::DataType::kINT4:
            data_type_size = sizeof(float);
            break;
    }

    return data_type_size;
}

