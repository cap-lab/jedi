#include <libconfig.h++>
#include <cstring>
#include <fstream>
#include <sstream>
#include <math.h>
#include <limits>

#include "tensorrt_network.h"

#include "basic_onnx_application.h"

void BasicOnnxApplication::readOnnxFilePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["onnx_file_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		onnxAppConfig.onnx_file_path = data.c_str();

		std::cerr<<"onnx_file_path: "<<onnxAppConfig.onnx_file_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'onnx_file_path' setting in configuration file." << std::endl;
        onnxAppConfig.onnx_file_path = "";
	}
}

void BasicOnnxApplication::readQuantizedOnnxFilePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["quantized_onnx_file_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		onnxAppConfig.quantized_onnx_file_path = data.c_str();

		std::cerr<<"quantized_onnx_file_path: "<<onnxAppConfig.quantized_onnx_file_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'quantized_onnx_file_path' setting in configuration file." << std::endl;
        onnxAppConfig.quantized_onnx_file_path = "";
	}
}

void BasicOnnxApplication::readSecondaryOnnxFilePath(libconfig::Setting &setting)
{
	try
	{
		const char *tmp = setting["secondary_onnx_file_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		onnxAppConfig.secondary_onnx_file_path = data.c_str();

		std::cerr << "secondary_onnx_file_path: " << onnxAppConfig.secondary_onnx_file_path << std::endl;
	}
	catch (const libconfig::SettingNotFoundException &nfex)
	{
		// Since secondary onnx file path is an optional parameter, do not print waring messages when the option is not set.
		// std::cerr << "No 'secondary_onnx_file_path' setting in configuration file." << std::endl;
		onnxAppConfig.secondary_onnx_file_path = "";
	}
}

void BasicOnnxApplication::readSecondaryCalibrationTablePath(libconfig::Setting &setting)
{
	try
	{
		const char *tmp = setting["secondary_calib_table"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		onnxAppConfig.secondary_calib_table = data.c_str();

		std::cerr << "secondary_calib_table: " << onnxAppConfig.secondary_calib_table << std::endl;
	}
	catch (const libconfig::SettingNotFoundException &nfex)
	{
		// Since secondary calibration table path is an optional parameter, do not print waring messages when the option is not set.
		// std::cerr << "No 'secondary_calib_table' setting in configuration file." << std::endl;
		onnxAppConfig.secondary_calib_table = "";
	}
}

void BasicOnnxApplication::readOptimizationProfileFilePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["optimization_cfg_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		onnxAppConfig.optimization_cfg_path = data.c_str();
		std::cerr<<"optimization_cfg_path: "<<onnxAppConfig.optimization_cfg_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'optimization_cfg_path' setting in configuration file." << std::endl;
		//exit(EXIT_FAILURE);
	}
}


void BasicOnnxApplication::readCustomOptions(libconfig::Setting &setting)
{
	readOnnxFilePath(setting);
    readQuantizedOnnxFilePath(setting);
    readOptimizationProfileFilePath(setting);
	readSecondaryOnnxFilePath(setting);
	readSecondaryCalibrationTablePath(setting);
}
