
#ifndef CNET_APPLICATION_H_
#define CNET_APPLICATION_H_


#include "variable.h"

#include <opencv2/opencv.hpp>

#include "image_dataset.h"

#include "inference_application.h"

#include "config.h"
#include "cuda.h"

typedef struct _CenternetAppConfig {
	std::string image_path;
	std::string calib_image_path;
	int calib_images_num;
	std::string name_path;
} CenternetAppConfig;

typedef struct _CenterPostProcessingSharedData {
	float *scores;
	int   *clses;
	float *bbx0;
	float *bby0;
	float *bbx1;
	float *bby1;
} CenterPostProcessingSharedData;

typedef struct _CenterPostProcessingGPUData {
	CenterPostProcessingSharedData shared;
	int * topk_inds;
	float * topk_ys;
	float * topk_xs;
	int * inttopk_ys;
	int * inttopk_xs;
	float *src_out;
	int *ids_out;
} CenterPostProcessingGPUData;


#define K_VALUE 100

class CenternetApplication : public IInferenceApplication {
	public:
		CenternetApplication() {
			mean << 0.408, 0.447, 0.47;
			stddev << 0.289, 0.274, 0.278;
		    dim_hm = tk::dnn::dataDim_t(1, 80, 128, 128, 1);
		    dim_wh = tk::dnn::dataDim_t(1, 2, 128, 128, 1);
		    dim_reg = tk::dnn::dataDim_t(1, 2, 128, 128, 1);

		    ids = cuda_make_int_array_host(dim_hm.c * dim_hm.h * dim_hm.w);
		    for(int i =0; i<dim_hm.c * dim_hm.h * dim_hm.w; i++){
		    	ids[i] = i;
		    }
		    //cudaHostGetDevicePointer(&ids_d, ids, 0);
		};
		~CenternetApplication();
		void initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void preprocessing(int thread_id, int sample_index, int batch_index, IN OUT float *input_buffer) override;
		void initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number) override;
		void postprocessing(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch, IN OUT int *buffer_occupied) override;
		tk::dnn::Network* createNetwork(ConfigInstance *basic_config_data) override;
		void referNetworkRTInfo(int device_id, tk::dnn::NetworkRT *networkRT) override;
		void readCustomOptions(libconfig::Setting &setting) override;

	private:
		cv::Mat dst = cv::Mat(cv::Size(2,3), CV_32F);
		cv::Mat dst2 = cv::Mat(cv::Size(2,3), CV_32F);
		cv::Vec<float, 3> mean;
		cv::Vec<float, 3> stddev;
		int *ids;
		int *ids_d;
		std::vector<CenterPostProcessingSharedData *> cpuDataList;
		std::vector<CenterPostProcessingGPUData *> gpuDataList;

	    std::vector<cudaStream_t> post_streams;

	    tk::dnn::dataDim_t dim_hm;
	    tk::dnn::dataDim_t dim_wh;
	    tk::dnn::dataDim_t dim_reg;
		CenternetAppConfig centernetAppConfig;
		InputDim input_dim;
		//bool letter_box;
		ImageDataset *dataset;
		//std::vector<Detection *> dets_vec;
		//std::vector<std::vector<int>> detection_num_vec;
		std::string network_name;
		std::vector<float *> target_coords_vec;
		std::vector<Detection *> dets_vec;
		std::vector<std::vector<int>> detection_num_vec;

		void readImagePath(libconfig::Setting &setting);
		void readCalibImagePath(libconfig::Setting &setting);
		void readCalibImagesNum(libconfig::Setting &setting);
		void readNamePath(libconfig::Setting &setting);
		void printBox(int sample_index, int batch, Detection *dets, std::vector<int> detections_num);
};

#endif


