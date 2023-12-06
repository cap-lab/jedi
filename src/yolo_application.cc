#include <libconfig.h++>
#include <cstring>
#include <sstream>
#include <math.h>

#include <opencv2/opencv.hpp>

#include <tkDNN/DarknetParser.h>

#include "image_opencv.h"

#include "box.h"
#include "yolo_wrapper.h"
#include "region_wrapper.h"

#include "tkdnn_network.h"

#include "yolo_application.h"

#define NMS 0.45

REGISTER_JEDI_APPLICATION(YoloApplication);

void YoloApplication::readCfgPath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["cfg_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		yoloAppConfig.cfg_path = data.c_str();

		std::cerr<<"cfg_path: "<<yoloAppConfig.cfg_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'cfg_path' setting in configuration file." << std::endl;
	}
}

void YoloApplication::readImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		yoloAppConfig.image_path = data.c_str();

		std::cerr<<"image_path: "<<yoloAppConfig.image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'image_path' setting in configuration file." << std::endl;
	}
}

void YoloApplication::readCalibImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["calib_image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		yoloAppConfig.calib_image_path = data.c_str();

		std::cerr<<"calib_image_path: "<<yoloAppConfig.calib_image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_image_path' setting in configuration file." << std::endl;
	}
}

void YoloApplication::readCalibImagesNum(libconfig::Setting &setting){
	try {
		const char *data = setting["calib_images_num"];
		yoloAppConfig.calib_images_num = atoi(data);

		std::cerr<<"calib_images_num: "<<yoloAppConfig.calib_images_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_images_num' setting in configuration file." <<std::endl;
	}
}

void YoloApplication::readNamePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["name_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;

		yoloAppConfig.name_path = data.c_str();

		std::cerr<<"name_path: "<<yoloAppConfig.name_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'name_path' setting in configuration file." << std::endl;
	}
}

void YoloApplication::readOpenCVParallelNum(libconfig::Setting &setting) {
	try{	
		const char *data = setting["opencv_parallel_num"];
		yoloAppConfig.opencv_parallel_num = atoi(data);

		std::cerr<<"opencv_parallel_num: "<<yoloAppConfig.opencv_parallel_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'opencv_parallel_num' setting in configuration file. Set -1 as a Default." << std::endl;
		yoloAppConfig.opencv_parallel_num = -1;
	}
}



void YoloApplication::readCustomOptions(libconfig::Setting &setting)
{
	readCfgPath(setting);
	readImagePath(setting);
	readCalibImagePath(setting);
	readCalibImagesNum(setting);
	readNamePath(setting);
	readOpenCVParallelNum(setting);
}

IJediNetwork *YoloApplication::createNetwork(ConfigInstance *basic_config_data)
{
	std::string cfg_path = yoloAppConfig.cfg_path;
	std::string name_path = yoloAppConfig.name_path;
	std::string bin_path(basic_config_data->bin_path);
	std::string wgs_path  = bin_path + "/layers";

	TkdnnNetwork *jedi_network = new TkdnnNetwork();

	jedi_network->net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);

	letter_box = jedi_network->net->letterBox;
	input_dim.width = jedi_network->net->input_dim.w;
	input_dim.height = jedi_network->net->input_dim.h;
	input_dim.channel = jedi_network->net->input_dim.c;

	jedi_network->net->fileImgList = yoloAppConfig.calib_image_path;
	jedi_network->net->num_calib_images = yoloAppConfig.calib_images_num;

	for (int iter = 0 ; iter < jedi_network->net->num_layers; iter++) {
		if(jedi_network->net->layers[iter]->getLayerType() == LAYER_YOLO) {
			YoloData yolo;
			tk::dnn::Yolo *yoloTKDNN = (tk::dnn::Yolo *) jedi_network->net->layers[iter];
			yolo.n_masks = yoloTKDNN->n_masks;
			yolo.bias = yoloTKDNN->bias_h;
			yolo.mask = yoloTKDNN->mask_h;
			yolo.new_coords = yoloTKDNN->new_coords;
			yolo.nms_kind = (tk::dnn::Yolo::nmsKind_t) yoloTKDNN->nsm_kind;
			yolo.nms_thresh = yoloTKDNN->nms_thresh;
			yolo.height = yoloTKDNN->input_dim.h;
			yolo.width = yoloTKDNN->input_dim.w;
			yolo.channel = yoloTKDNN->input_dim.c;
			yolo.scale_x_y = yoloTKDNN->scaleXY;
			yolo.num = yoloTKDNN->num;

			yolos.push_back(yolo);
		}
	}

	return jedi_network;
}

void YoloApplication::initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	this->network_name = network_name;
	dataset = new ImageDataset(yoloAppConfig.image_path);
	result_format = new COCOFormat();

	if(yoloAppConfig.opencv_parallel_num >= 0) {
		cv::setNumThreads(0);
	}
}



void YoloApplication::preprocessing(int thread_id, int input_tensor_index, const char *input_name, int sample_index, int batch_index, IN OUT float *input_buffer)
{
	int image_index = (sample_index + batch_index) % dataset->getSize();

	ImageData *image_data = dataset->getData(image_index);
	int orignal_width = 0;
	int original_height = 0;

	if(letter_box == true)
		loadImageLetterBox((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);
	else
		loadImageResize((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);

	image_data->width = orignal_width;
	image_data->height = original_height;
}


void YoloApplication::regionLayerDetect(int sampleIndex, int batch, float *output, Detection *dets, std::vector<int> &detections_num)
{
	int in_width = 0, in_height = 0;

	in_width = input_dim.width / 32;
	in_height = input_dim.height / 32;

	for(int i = 0; i < batch; i++) {
		ImageData *data = dataset->getData(sampleIndex * batch + i);
		int orig_width = data->width;
		int orig_height = data->height;
		int count = 0;
		get_region_detections(&output[i * in_width * in_height * NUM_ANCHOR * (NUM_CLASSES + 5)], CONFIDENCE_THRESH, input_dim, &dets[i * NBOXES], &count, orig_width, orig_height);
		do_nms_sort(&dets[i * NBOXES], count, NMS);
		detections_num[i] = count;
	}
}

void YoloApplication::yoloLayerDetect(int sampleIndex, int batch, float **output_buffers, int output_num, Detection *dets, std::vector<int> &detections_num)
{
	int detection_num = 0;
	int output_size = 0;
	int yolo_num = yolos.size();

	for (int iter1 = 0; iter1 < batch; iter1++) {
		ImageData *data = dataset->getData(sampleIndex * batch + iter1);
		int orig_width = data->width;
		int orig_height = data->height;

		detection_num = 0;

		for(int iter2 = 0; iter2 < yolo_num; iter2++) {
			int w = yolos[iter2].width;
			int h = yolos[iter2].height;
			int c = yolos[iter2].channel;

			output_size = w * h * c;
			yolo_computeDetections(output_buffers[iter2] + output_size * iter1, &dets[iter1 * NBOXES], &detection_num, w, h, c, CONFIDENCE_THRESH, yolos[iter2], orig_width, orig_height, input_dim.width, input_dim.height, letter_box);
		}
		yolo_mergeDetections(&dets[iter1 * NBOXES], detection_num, yolos[0].nms_thresh, yolos[0].nms_kind);
		detections_num[iter1] = detection_num;
	}
}

void YoloApplication::detectBox(float **output_buffers, int output_num, int sampleIndex, int batch, Detection *dets, std::vector<int> &detections_num)
{
	if(network_name == NETWORK_YOLOV2 || network_name == NETWORK_YOLOV2TINY || network_name == NETWORK_DENSENET) {
		regionLayerDetect(sampleIndex, batch, output_buffers[0], dets, detections_num);
	}
	else {
		yoloLayerDetect(sampleIndex, batch, output_buffers, output_num, dets, detections_num);
	}
}


void YoloApplication::printBox(int sample_index, int batch, Detection *dets, std::vector<int> detections_num)
{
	for(int iter1 = 0; iter1 < batch; iter1++) {
		int image_index = (sample_index * batch + iter1) % dataset->getSize();
		ImageData *data = dataset->getData(image_index);
		char *path = (char *)(data->path.c_str());

		result_format->detectCOCO(&dets[iter1 * NBOXES], detections_num[iter1], image_index, data->width, data->height, input_dim.width, input_dim.height, path);
	}
}


void YoloApplication::initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	setBiases(network_name);
	for (int i = 0 ; i < thread_number ; i++ ) {
		Detection *dets;
		allocateDetectionBox(maximum_batch_size, &dets);
		dets_vec.push_back(dets);

		this->detection_num_vec.push_back(std::vector<int>(maximum_batch_size, 0));
	}
}

void YoloApplication::postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch)
{
	detectBox(output_buffers, output_num, sample_index, batch, dets_vec[thread_id], detection_num_vec[thread_id]);

}

void YoloApplication::postprocessing2(int thread_id, int sample_index, int batch) {
	printBox(sample_index, batch, dets_vec[thread_id], detection_num_vec[thread_id]);
}

void YoloApplication::writeResultFile(std::string result_file_name) {
	result_format->writeResultFile(result_file_name);
}

YoloApplication::~YoloApplication()
{
	int batch = this->detection_num_vec[0].size();

	while(dets_vec.size() > 0)
	{
		Detection *det = dets_vec.back();
		deallocateDetectionBox(batch * NBOXES, det);
		dets_vec.pop_back();
	}
	yolos.clear();
	delete dataset;
	delete result_format;
}
