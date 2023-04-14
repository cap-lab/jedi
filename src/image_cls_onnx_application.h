
#ifndef YOLO_ONNX_APPLICATION_H_
#define YOLO_ONNX_APPLICATION_H_

#include "variable.h"
#include "imagenet_format.h"

#include "inference_application.h"

#include "config.h"

typedef struct _ImageClsOnnxAppConfig {
	std::string onnx_file_path;
	std::string calib_image_path;
	std::string image_path;
	std::string label_path;
	int calib_images_num;
	int opencv_parallel_num;
} ImageClsOnnxAppConfig;


class ImageClsOnnxApplication : public IInferenceApplication {
	public:
		ImageClsOnnxApplication() {};
		~ImageClsOnnxApplication();
		void initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void preprocessing(int thread_id, int sample_index, int batch_index, IN OUT float *input_buffer) override;
		void initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch) override;
		void postprocessing2(int thread_id, int sample_index, int batch) override;
		void readCustomOptions(libconfig::Setting &setting) override;
		//tk::dnn::Network* createNetwork(ConfigInstance *basic_config_data) override;
		IJediNetwork* createNetwork(ConfigInstance *basic_config_data) override;

	private:
		ImageClsOnnxAppConfig imageClsOnnxAppConfig;
		InputDim input_dim;
		ImageDataset *dataset;
		ImagenetFormat *result_format;
		std::string network_name;
		int class_num;
		std::vector<std::string> labels;

		void readOnnxFilePath(libconfig::Setting &setting);
		void readCalibImagePath(libconfig::Setting &setting);
		void readCalibImagesNum(libconfig::Setting &setting);
		void readImagePath(libconfig::Setting &setting);
		void readLabelPath(libconfig::Setting &setting);
		void readOpenCVParallelNum(libconfig::Setting &setting);

		char* nolibStrStr(const char *s1, const char *s2);
		int generateTruths(std::string path);
		void writeResultFile(std::string result_file_name);

};

#endif


