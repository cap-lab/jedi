#include <libconfig.h++>
#include <cstring>
#include <sstream>
#include <math.h>

#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>


#include "image_opencv.h"


#include "box.h"
#include "yolo_wrapper.h"
#include "region_wrapper.h"

#include "tensorrt_network.h"

#include "yolo_onnx_application.h"


#include <tkDNN/tkdnn.h>
#include <tkDNN/Int8BatchStream.h>
#include <tkDNN/Int8Calibrator.h>

#define NMS 0.45


using namespace nvinfer1;
using namespace nvonnxparser; 

REGISTER_JEDI_APPLICATION(YoloOnnxApplication);


#ifndef FatalError
#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}
#endif 


class OnnxParserLogger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        //if (severity <= Severity::kWARNING)
		std::cout <<"TENSORRT ONNX LOG: "<< msg << std::endl;
    }
} onnx_logger;

static float g_bias[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 };
static float g_mask[3][3] = { { 6, 7, 8}, {3, 4, 5}, {0, 1, 2} };

// static inline float logisticActivate(float x){return 1.f/(1.f + expf(-x));}

void YoloOnnxApplication::readOnnxFilePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["onnx_file_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		yoloOnnxAppConfig.onnx_file_path = data.c_str();

		std::cerr<<"onnx_file_path: "<<yoloOnnxAppConfig.onnx_file_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'onnx_file_path' setting in configuration file." << std::endl;
	}
}

void YoloOnnxApplication::readCalibImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["calib_image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		yoloOnnxAppConfig.calib_image_path = data.c_str();

		std::cerr<<"calib_image_path: "<<yoloOnnxAppConfig.calib_image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_image_path' setting in configuration file." << std::endl;
	}
}

void YoloOnnxApplication::readCalibImagesNum(libconfig::Setting &setting){
	try {
		const char *data = setting["calib_images_num"];
		yoloOnnxAppConfig.calib_images_num = atoi(data);

		std::cerr<<"calib_images_num: "<<yoloOnnxAppConfig.calib_images_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_images_num' setting in configuration file." <<std::endl;
	}
}



void YoloOnnxApplication::readImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		yoloOnnxAppConfig.image_path = data.c_str();

		std::cerr<<"image_path: "<<yoloOnnxAppConfig.image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'image_path' setting in configuration file." << std::endl;
	}
}

void YoloOnnxApplication::readNamePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["name_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;

		yoloOnnxAppConfig.name_path = data.c_str();

		std::cerr<<"name_path: "<<yoloOnnxAppConfig.name_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'name_path' setting in configuration file." << std::endl;
	}
}

void YoloOnnxApplication::readOpenCVParallelNum(libconfig::Setting &setting) {
	try{	
		const char *data = setting["opencv_parallel_num"];
		yoloOnnxAppConfig.opencv_parallel_num = atoi(data);

		std::cerr<<"opencv_parallel_num: "<<yoloOnnxAppConfig.opencv_parallel_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'opencv_parallel_num' setting in configuration file. Set -1 as a Default." << std::endl;
		yoloOnnxAppConfig.opencv_parallel_num = -1;
	}
}



void YoloOnnxApplication::readCustomOptions(libconfig::Setting &setting)
{
	readOnnxFilePath(setting);
	readImagePath(setting);
	readNamePath(setting);
	readOpenCVParallelNum(setting);
	readCalibImagePath(setting);
	readCalibImagesNum(setting);
}

IJediNetwork *YoloOnnxApplication::createNetwork(ConfigInstance *basic_config_data)
{
	std::string calib_table = basic_config_data->calib_table;
	std::string name_path = yoloOnnxAppConfig.name_path;
	TensorRTNetwork *jedi_network = new TensorRTNetwork();

	jedi_network->builder = createInferBuilder(onnx_logger);

	uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
	jedi_network->network =  jedi_network->builder->createNetworkV2(flag);
	jedi_network->onnx_file_path = yoloOnnxAppConfig.onnx_file_path;

	IParser* parser = createParser(*(jedi_network->network), onnx_logger);

	// TODO: onnx file path
	parser->parseFromFile(yoloOnnxAppConfig.onnx_file_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << "TENSORRT ONNX ERROR: "  << parser->getError(i)->desc() << std::endl;
	}

	if(parser->getNbErrors() > 0) {
		FatalError("Onnx parsing failed");
	}

	// Printing network inputs with dimensions
	/*int nNumOfInput = jedi_network->network->getNbInputs();
	for(int index = 0 ; index < nNumOfInput ; index++) {
		ITensor *tensor = jedi_network->network->getInput(index);
		Dims tensor_dim = tensor->getDimensions();
		for(int index2 = 0; index2 < tensor_dim.nbDims ; index2++) {
			std::cout << tensor_dim.d[index2] << std::endl;
		}

	}*/

	letter_box = false;
	ITensor *tensor = jedi_network->network->getInput(0);
	Dims tensor_dim = tensor->getDimensions();
	input_dim.channel = tensor_dim.d[1];
	input_dim.width = tensor_dim.d[2];
	input_dim.height = tensor_dim.d[3];
	dataDim_t dim(tensor_dim.d[0],tensor_dim.d[1], tensor_dim.d[2], tensor_dim.d[3]);
	BatchStream *calibrationStream = new BatchStream(dim, 1, yoloOnnxAppConfig.calib_images_num, yoloOnnxAppConfig.calib_image_path);
	Int8EntropyCalibrator *calibrator = new Int8EntropyCalibrator(*calibrationStream, 1, calib_table , "data");
	jedi_network->calibrator = calibrator;

	;

	{
		YoloData yolo;

		yolo.n_masks = 3;
		yolo.bias = g_bias;
		yolo.mask = g_mask[0];
		yolo.new_coords = 0;
		yolo.nms_kind = (tk::dnn::Yolo::nmsKind_t) 0;
		yolo.nms_thresh = 0.45;
		yolo.height = jedi_network->network->getOutput(0)->getDimensions().d[2]; // image height / 32
		yolo.width = jedi_network->network->getOutput(0)->getDimensions().d[3]; // image width / 32
		yolo.channel = jedi_network->network->getOutput(0)->getDimensions().d[1];
		yolo.scale_x_y = 1;
		yolo.num = 3;

		yolos.push_back(yolo);
	}

	{
		YoloData yolo;

		yolo.n_masks = 3;
		yolo.bias = g_bias;
		yolo.mask = g_mask[1];
		yolo.new_coords = 0;
		yolo.nms_kind = (tk::dnn::Yolo::nmsKind_t) 0;
		yolo.nms_thresh = 0.45;
		yolo.height = jedi_network->network->getOutput(1)->getDimensions().d[2];  // image height / 16
		yolo.width = jedi_network->network->getOutput(1)->getDimensions().d[3]; // image width / 16
		yolo.channel = jedi_network->network->getOutput(1)->getDimensions().d[1];
		yolo.scale_x_y = 1;
		yolo.num = 3;

		yolos.push_back(yolo);
	}

	{
		YoloData yolo;

		yolo.n_masks = 3;
		yolo.bias = g_bias;
		yolo.mask = g_mask[2];
		yolo.new_coords = 0;
		yolo.nms_kind = (tk::dnn::Yolo::nmsKind_t) 0;
		yolo.nms_thresh = 0.45;
		yolo.height = jedi_network->network->getOutput(2)->getDimensions().d[2];  // image height / 8
		yolo.width = jedi_network->network->getOutput(2)->getDimensions().d[3];  // image width / 8
		yolo.channel = jedi_network->network->getOutput(2)->getDimensions().d[1];
		yolo.scale_x_y = 1;
		yolo.num = 3;

		yolos.push_back(yolo);
	}

	return jedi_network;
}

void YoloOnnxApplication::initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	this->network_name = network_name;
	dataset = new ImageDataset(yoloOnnxAppConfig.image_path);

	if(yoloOnnxAppConfig.opencv_parallel_num >= 0) {
		cv::setNumThreads(0);
	}
}



void YoloOnnxApplication::preprocessing(int thread_id, int sample_index, int batch_index, IN OUT float *input_buffer)
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


void YoloOnnxApplication::regionLayerDetect(int sampleIndex, int batch, float *output, Detection *dets, std::vector<int> &detections_num)
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

void YoloOnnxApplication::yoloLayerDetect(int sampleIndex, int batch, float **output_buffers, int output_num, Detection *dets, std::vector<int> &detections_num)
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

void YoloOnnxApplication::activateArrayLogistic(float *x, int n)
{
	int iter1 = 0;
	// #pragma omp parallel for
	for (iter1 = 0; iter1 < n; iter1++) {
		// x[iter1] = logisticActivate(x[iter1]);
		x[iter1] = 1.f/(1.f + expf(-x[iter1]));
	}
}

void YoloOnnxApplication::scalAddCPU(int N, float ALPHA, float BETA, float *X, int INCX)
{
	int iter1 = 0;
	for (iter1 = 0; iter1 < N; ++iter1) X[iter1*INCX] = X[iter1*INCX] * ALPHA + BETA;
}

void YoloOnnxApplication::forwardYoloLayer(float **output_buffers, int batch)
{
    int yolo_num = yolos.size();

    for (int iter1 = 0; iter1 < yolo_num; iter1++) {
        int num = yolos[iter1].num;
        int w = yolos[iter1].width;
        int h = yolos[iter1].height;
        int c = yolos[iter1].channel;
        int new_coords = yolos[iter1].new_coords;
        float scale_x_y = yolos[iter1].scale_x_y;

        for (int iter2 = 0; iter2 < batch; iter2++) {
            for (int iter3 = 0; iter3 < num; iter3++) {
                int bbox_index = entry_yolo_index(iter2, iter3*w*h, 0, w, h, c);    

                if(!new_coords) {
                    activateArrayLogistic(output_buffers[iter1] + bbox_index, 2*w*h);   
                    int obj_index = entry_yolo_index(iter2, iter3*w*h, 4, w, h, c); 
                    activateArrayLogistic(output_buffers[iter1] + obj_index, (1+NUM_CLASSES)*w*h);  
                }   
                scalAddCPU(2*w*h, scale_x_y, -0.5*(scale_x_y-1), output_buffers[iter1] + bbox_index, 1); 
            }   
        }   
    }   
}

void YoloOnnxApplication::detectBox(float **output_buffers, int output_num, int sampleIndex, int batch, Detection *dets, std::vector<int> &detections_num)
{
	if(network_name == NETWORK_YOLOV2 || network_name == NETWORK_YOLOV2TINY || network_name == NETWORK_DENSENET) {
		regionLayerDetect(sampleIndex, batch, output_buffers[0], dets, detections_num);
	}
	else {
		// TODO: for ONNX
		forwardYoloLayer(output_buffers, batch);
		yoloLayerDetect(sampleIndex, batch, output_buffers, output_num, dets, detections_num);
	}
}


void YoloOnnxApplication::printBox(int sample_index, int batch, Detection *dets, std::vector<int> detections_num)
{
	for(int iter1 = 0; iter1 < batch; iter1++) {
		int image_index = (sample_index * batch + iter1) % dataset->getSize();
		ImageData *data = dataset->getData(image_index);
		char *path = (char *)(data->path.c_str());


		detectCOCO(&dets[iter1 * NBOXES], detections_num[iter1], image_index, data->width, data->height, input_dim.width, input_dim.height, path);
	}
}


void YoloOnnxApplication::initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	setBiases(network_name);
	for (int i = 0 ; i < thread_number ; i++ ) {
		Detection *dets;
		allocateDetectionBox(maximum_batch_size, &dets);
		dets_vec.push_back(dets);

		this->detection_num_vec.push_back(std::vector<int>(maximum_batch_size, 0));
	}
}

void YoloOnnxApplication::postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch)
{
	detectBox(output_buffers, output_num, sample_index, batch, dets_vec[thread_id], detection_num_vec[thread_id]);
}

void YoloOnnxApplication::postprocessing2(int thread_id, int sample_index, int batch) {
	printBox(sample_index, batch, dets_vec[thread_id], detection_num_vec[thread_id]);
}

YoloOnnxApplication::~YoloOnnxApplication()
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
}
