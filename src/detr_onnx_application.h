
#ifndef DETR_ONNX_APPLICATION_H_
#define DETR_ONNX_APPLICATION_H_

#include "variable.h"
#include "coco_format.h"
#include "inference_application.h"
#include "config.h"
#include "yolo_wrapper.h"

#define DETR_MAX_BOXES_SINGLE_IMAGE 100
#define DETR_NUM_CLASSES 92

typedef struct _DETROnnxAppConfig {
	std::string onnx_file_path;
	std::string calib_image_path;
	std::string image_path;
	std::string name_path;
	int calib_images_num;
	int opencv_parallel_num;
} DETROnnxAppConfig;


typedef struct _DETRData {
	// float *mask;
	// int n_masks;
	// float *bias;
	int new_coords;
	double nms_thresh;
	// tk::dnn::Yolo::nmsKind_t nms_kind;
	int height;
	int width;
	int channel;
	float scale_x_y;
	int num;
} DETRData;



class DETROnnxApplication : public IInferenceApplication {
	public:
		DETROnnxApplication() {};
		~DETROnnxApplication();
		void initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void preprocessing(int thread_id, int input_tensor_index, const char *input_name, int sample_index, int batch_index, IN OUT float *input_buffer) override;
		void initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch) override;
        void postprocessing2(int thread_id, int sample_index, int batch);
        void readCustomOptions(libconfig::Setting &setting) override;
        //tk::dnn::Network* createNetwork(ConfigInstance *basic_config_data) override;
		IJediNetwork* createNetwork(ConfigInstance *basic_config_data) override;

	private:
		DETROnnxAppConfig detrOnnxAppConfig;
		InputDim input_dim;
		bool letter_box;
		ImageDataset *dataset;
		COCOFormat *result_format;
		std::vector<Detection *> dets_vec;
		std::vector<std::vector<int>> detection_num_vec;
		std::string network_name;
		int num_detections =  100;
		int num_classes = 92;

		void readOnnxFilePath(libconfig::Setting &setting);
		void readCalibImagePath(libconfig::Setting &setting);
		void readCalibImagesNum(libconfig::Setting &setting);
		void readImagePath(libconfig::Setting &setting);
		void readNamePath(libconfig::Setting &setting);
		void readOpenCVParallelNum(libconfig::Setting &setting);

        void computeConfidenceAndLabels(float *logit, float &confidence, int &label);
        void softmax(float *logit);
        int get_coco_image_id(char *filename);

		void writeResultFile(std::string result_file_name);
};

#endif


