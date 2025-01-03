
#ifndef BASIC_ONNX_APPLICATION_H_
#define BASIC_ONNX_APPLICATION_H_

#include "variable.h"

#include "inference_application.h"

#include "config.h"


typedef struct _BasicOnnxAppConfig {
	std::string onnx_file_path;
	std::string quantized_onnx_file_path;
	std::string optimization_cfg_path;
} BasicOnnxAppConfig;


class BasicOnnxApplication : public IInferenceApplication {
	public:
		BasicOnnxApplication() {};
		~BasicOnnxApplication() {};
		void readCustomOptions(libconfig::Setting &setting) override;
	protected:
		BasicOnnxAppConfig onnxAppConfig;
	private:
		void readOnnxFilePath(libconfig::Setting &setting);
		void readQuantizedOnnxFilePath(libconfig::Setting &setting);
		void readOptimizationProfileFilePath(libconfig::Setting &setting);
};

#endif


