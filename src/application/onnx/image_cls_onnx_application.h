
#ifndef IMAGE_CLS_ONNX_APPLICATION_H_
#define IMAGE_CLS_ONNX_APPLICATION_H_

#include "variable.h"
#include "imagenet_format.h"

#include "inference_application.h"

#include "config.h"

#include "basic_onnx_application.h"


typedef struct _ImageClsOnnxAppConfig {
	std::string calib_image_path;
	std::string image_path;
	std::string label_path;
	int calib_images_num;
	int opencv_parallel_num;
	ImagePreprocessingOption preprocessing_option;
	ResizeInterpolationOption interpolation;
	float mean[IMAGE_COLOR_NUM];
	float std[IMAGE_COLOR_NUM];
	int crop_base_size;
} ImageClsOnnxAppConfig;


class ImageClsOnnxApplication : public BasicOnnxApplication {
	public:
		ImageClsOnnxApplication() {};
		~ImageClsOnnxApplication();
		void initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void preprocessing(int thread_id, int input_tensor_index, const char *input_name, int sample_index, int batch_index, IN OUT float *input_buffer) override;
		void initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void postprocessing1(int thread_id, int sample_index, IN void **output_buffers, int output_num, int batch) override;
		void postprocessing2(int thread_id, int sample_index, int batch) override;
		void readCustomOptions(libconfig::Setting &setting) override;
		IJediNetwork* createNetwork(ConfigInstance *basic_config_data) override;

	private:
		ImageClsOnnxAppConfig imageClsOnnxAppConfig;
		InputDim input_dim;
		ImageDataset *dataset = nullptr;
		ImagenetFormat *result_format = nullptr;
		std::string network_name;
		int class_num;
		std::vector<std::string> labels;

		void readCalibImagePath(libconfig::Setting &setting);
		void readCalibImagesNum(libconfig::Setting &setting);
		void readImagePath(libconfig::Setting &setting);
		void readLabelPath(libconfig::Setting &setting);
		void readOpenCVParallelNum(libconfig::Setting &setting);
		void readImagePreprocessingOption(libconfig::Setting &setting);
		void readImageNormalizeMeanOption(libconfig::Setting &setting);
		void readImageNormalizeStdOption(libconfig::Setting &setting);
		void readInterpolationOption(libconfig::Setting &setting);
		void readCropSizeOption(libconfig::Setting &setting);

		char* nolibStrStr(const char *s1, const char *s2);
		int generateTruths(std::string path);
		void writeResultFile(std::string result_file_name);
		void softmax(float *logit);

};

#endif


