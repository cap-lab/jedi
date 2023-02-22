
#ifndef TENSORRT_NETWORK_H_
#define TENSORRT_NETWORK_H_

//#include <tkDNN/tkdnn.h>

#include <NvInfer.h>

#include "config_data.h"

#include "jedi_network.h"

class TensorRTNetwork : public IJediNetwork {
	public:
		TensorRTNetwork() {};
		nvinfer1::INetworkDefinition* network;
		nvinfer1::IBuilder * builder;
		std::string onnx_file_path;
		//void createNetwork() override;
		void printNetwork();
	private:
};


#endif
