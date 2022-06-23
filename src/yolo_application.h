
#ifndef YOLO_APPLICATION_H_
#define YOLO_APPLICATION_H_

#include "variable.h"
#include "image_dataset.h"

#include "inference_application.h"

#include "config.h"

#include "yolo_wrapper.h"

typedef struct _YoloAppConfig {
	std::string cfg_path;
	std::string image_path;
	std::string calib_image_path;
	int calib_images_num;
	std::string name_path;
	int opencv_parallel_num;
} YoloAppConfig;


class YoloApplication : public IInferenceApplication {
	public:
		YoloApplication() {};
		~YoloApplication();
		void initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void preprocessing(int thread_id, int sample_index, int batch_index, IN OUT float *input_buffer) override;
		void initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch) override;
		void postprocessing2(int thread_id, int sample_index, int batch) override;
		tk::dnn::Network* createNetwork(ConfigInstance *basic_config_data) override;
		void referNetworkRTInfo(int device_id, tk::dnn::NetworkRT *networkRT) override;
		void readCustomOptions(libconfig::Setting &setting) override;

	private:
		YoloAppConfig yoloAppConfig;
		std::vector<YoloData> yolos;
		InputDim input_dim;
		bool letter_box;
		ImageDataset *dataset;
		std::vector<Detection *> dets_vec;
		std::vector<std::vector<int>> detection_num_vec;
		std::string network_name;

		void readCfgPath(libconfig::Setting &setting);
		void readImagePath(libconfig::Setting &setting);
		void readCalibImagePath(libconfig::Setting &setting);
		void readCalibImagesNum(libconfig::Setting &setting);
		void readNamePath(libconfig::Setting &setting);
		void readOpenCVParallelNum(libconfig::Setting &setting);
		void regionLayerDetect(int sampleIndex, int batch, float *output_buffer, Detection *dets, std::vector<int> &detections_num);
		void yoloLayerDetect(int sampleIndex, int batch, float **output_buffers, int output_num, Detection *dets, std::vector<int> &detections_num);
		void detectBox(float **output_buffers, int output_num, int sampleIndex, int batch, Detection *dets, std::vector<int> &detections_num);
		void printBox(int sample_index, int batch, Detection *dets, std::vector<int> detections_num);
};

#endif


