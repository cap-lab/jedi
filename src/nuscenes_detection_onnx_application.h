
#ifndef NUSCENES_DETECTION_ONNX_APPLICATION_H_
#define NUSCENES_DETECTION_ONNX_APPLICATION_H_

#include <map>

#include "variable.h"
#include "lidar_dataset.h"
#include "nuscenes_format.h"

#include "inference_application.h"

#include "config.h"


typedef struct _NuscenesDetectionOnnxAppConfig {
	std::string onnx_file_path;
	std::string optimization_cfg_path;
	std::string lidar_list_path;
	std::string calib_lidar_path;
	int calib_lidar_num;
} NuscenesDetectionOnnxAppConfig;


class NuscenesDetectionOnnxApplication : public IInferenceApplication {
	public:
		NuscenesDetectionOnnxApplication() {};
		~NuscenesDetectionOnnxApplication();
		void initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void preprocessing(int thread_id, int input_tensor_index, const char *input_name, int sample_index, int batch_index, IN OUT float *input_buffer) override;
		void initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch) override;
		void postprocessing2(int thread_id, int sample_index, int batch) override;
		void readCustomOptions(libconfig::Setting &setting) override;

		//tk::dnn::Network* createNetwork(ConfigInstance *basic_config_data) override;
		IJediNetwork* createNetwork(ConfigInstance *basic_config_data) override;

	private:
		NuscenesDetectionOnnxAppConfig nuscenesOnnxAppConfig;
		InputDim input_dim;
		LidarDataset *dataset;
		NuscenesFormat *result_format;
		std::string network_name;
		int class_num;
		std::vector<std::string> labels;
		std::vector<int> current_lidar_indexs;
		std::vector<float *> inputBuffers;
		std::vector<int> inputBufferSizes;

		std::vector<int *> indicesList;

		std::map<std::string, int> outputIndexMap;

		void readOnnxFilePath(libconfig::Setting &setting);
		void readOptimizationProfileFilePath(libconfig::Setting &setting);
		void readLidarListPath(libconfig::Setting &setting);
		void readCalibLidarPath(libconfig::Setting &setting);
		void readCalibLidarNum(libconfig::Setting &setting);

		void writeResultFile(std::string result_file_name);

};

#endif


