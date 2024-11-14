
#ifndef TENSORRT_NETWORK_H_
#define TENSORRT_NETWORK_H_

#include <NvInfer.h>

#include "config_data.h"

#include "jedi_network.h"

std::string convertLayerTypeToString(nvinfer1::ILayer *layer);

class TensorRTNetwork : public IJediNetwork {
	public:
		TensorRTNetwork() {};
		nvinfer1::INetworkDefinition* network = nullptr;
		nvinfer1::IBuilder *builder =nullptr;
		std::string onnx_file_path;
		std::string optimization_cfg_path;
		nvinfer1::IInt8Calibrator *calibrator = nullptr;
		//void createNetwork() override;
		void printNetwork();
	private:
};

class OnnxParserLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
			std::cout <<"TENSORRT ONNX LOG: "<< msg << std::endl;
    }
};

extern OnnxParserLogger onnx_logger;

#endif
