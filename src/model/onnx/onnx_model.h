#ifndef ONNX_MODEL_H_
#define ONNX_MODEL_H_

#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <NvInfer.h>
#include <NvOnnxParser.h>

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
		bool checkTensorIsUsedInNextStages(int device_id, nvinfer1::INetworkDefinition *network, int end_index, int layer_id, std::string tensor_name);
		void getIOTensorNamesOfLayer(nvinfer1::INetworkDefinition *network, int layer_id, std::vector<std::string>& tensor_name_vec, bool is_input);
		void surgeonOnnxByPolygraphy(int device_id, nvinfer1::INetworkDefinition *network, std::string model_name, std::string plan_file_name, int start_index, int end_index, int dequantize_skip_index);
		void separateOnnxFile(nvinfer1::INetworkDefinition *network, std::string model_name, std::string quantized_model_name, std::vector<std::string>& plan_file_name_vec);
		void initializeModel() override;
		void printModel() override;
		void finalizeModel() override;
	private:
		nvinfer1::IBuilderConfig* createEngineFromOnnxFile(int cur_iter, std::string onnx_file_name, bool is_quantized_onnx, nvinfer1::IBuilder* &builder, nvinfer1::INetworkDefinition* &network, nvonnxparser::IParser* &parser);
		bool serialize(const char *filename, nvinfer1::IHostMemory *ptr);
		std::vector<nvinfer1::IRuntime *> runtimes;
		void getModelFileName(int curr, std::string &plan_file_name, nvinfer1::INetworkDefinition *network, std::string postfix, bool for_rt_build, bool quantized_model_avaiable);
		void getOutputIndexOfStage(int device_id, nvinfer1::INetworkDefinition *network, int start_index, int end_index, std::vector<int>& output_index_vec, int dequantize_skip_index);
		void fillInputs(int device_id, nvinfer1::INetworkDefinition *network, int start_index, int end_index, std::vector<std::string>& input_name_vec, int dequantize_skip_index);
		void loadTimingCache(nvinfer1::IBuilderConfig* config, nvinfer1::ITimingCache* &cache);
		void saveTimingCache(nvinfer1::ITimingCache *cache);
		bool saveLayerInfoFile(std::string filename, const char *layerInfo);
};

#endif
