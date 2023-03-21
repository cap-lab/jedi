
#ifndef TENSORRT_NETWORK_H_
#define TENSORRT_NETWORK_H_

//#include <tkDNN/tkdnn.h>

#include <NvInfer.h>

#include "config_data.h"

#include "jedi_network.h"

class TensorRTNetwork : public IJediNetwork {
	public:
		TensorRTNetwork() {};
		nvinfer1::INetworkDefinition* network = nullptr;
		nvinfer1::IBuilder *builder =nullptr;
		std::string onnx_file_path;
		nvinfer1::IInt8Calibrator *calibrator = nullptr;
		//void createNetwork() override;
		void printNetwork();
	private:
};


#endif
