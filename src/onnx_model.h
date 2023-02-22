#ifndef ONNX_MODEL_H_
#define ONNX_MODEL_H_

#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include "variable.h"
#include "config.h"
#include "stage.h"

#include "inference_application.h"

#include "model.h"

#include "tensorrt_network.h"

class OnnxModel : public Model {
	public:
		OnnxModel(ConfigData *config_data, int instance_id, IInferenceApplication *app) : Model(config_data, instance_id, app) {};
		~OnnxModel() {};
		void initializeModel() override;
		void finalizeModel() override;
	private:
		bool serialize(const char *filename, nvinfer1::IHostMemory *ptr);
		std::vector<nvinfer1::IRuntime *> runtimes;
		void getModelFileName(int curr, std::string &plan_file_name, nvinfer1::INetworkDefinition *network);
};

#endif
