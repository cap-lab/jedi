#include <libconfig.h++>
#include <cstring>
#include <sstream>
#include <math.h>

#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "image_opencv.h"

#include "tensorrt_network.h"

#include "int8_image_batch_stream.h"
#include "int8_image_calibrator.h"

#include "yolo_end2end_onnx_application.h"

#define CALIBRATION_BATCH_SIZE (16)

using namespace nvinfer1;
using namespace nvonnxparser; 

REGISTER_JEDI_APPLICATION(YoloEnd2EndOnnxApplication);


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


// static inline float logisticActivate(float x){return 1.f/(1.f + expf(-x));}

void YoloEnd2EndOnnxApplication::readCalibImagePath(libconfig::Setting &setting) {
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


void YoloEnd2EndOnnxApplication::readCalibImagesNum(libconfig::Setting &setting){
	try {
		const char *data = setting["calib_images_num"];
		yoloOnnxAppConfig.calib_images_num = atoi(data);

		std::cerr<<"calib_images_num: "<<yoloOnnxAppConfig.calib_images_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_images_num' setting in configuration file." <<std::endl;
	}
}


void YoloEnd2EndOnnxApplication::readImagePath(libconfig::Setting &setting) {
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


void YoloEnd2EndOnnxApplication::readNamePath(libconfig::Setting &setting) {
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


void YoloEnd2EndOnnxApplication::readOpenCVParallelNum(libconfig::Setting &setting) {
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


void YoloEnd2EndOnnxApplication::readCustomOptions(libconfig::Setting &setting)
{
	BasicOnnxApplication::readCustomOptions(setting);
	readImagePath(setting);
	readNamePath(setting);
	readCalibImagePath(setting);
	readCalibImagesNum(setting);
}


IJediNetwork *YoloEnd2EndOnnxApplication::createNetwork(ConfigInstance *basic_config_data)
{
	std::string calib_table = basic_config_data->calib_table;
	TensorRTNetwork *jedi_network = new TensorRTNetwork();

	jedi_network->builder = createInferBuilder(onnx_logger);

#if NV_TENSORRT_MAJOR > 8
	uint32_t flag = 0;
#else
	uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); // deprecated in tensorrt 10
#endif
	jedi_network->network =  jedi_network->builder->createNetworkV2(flag);
	jedi_network->onnx_file_path = onnxAppConfig.onnx_file_path;

	IParser* parser = createParser(*(jedi_network->network), onnx_logger);

	// TODO: onnx file path
	parser->parseFromFile(onnxAppConfig.onnx_file_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << "TENSORRT ONNX ERROR: "  << parser->getError(i)->desc() << std::endl;
	}

	if(parser->getNbErrors() > 0) {
		FatalError("Onnx parsing failed");
	}

	jedi_network->optimization_cfg_path = onnxAppConfig.optimization_cfg_path;

	// Printing network inputs with dimensions
	/*int nNumOfInput = jedi_network->network->getNbInputs();
	for(int index = 0 ; index < nNumOfInput ; index++) {
		ITensor *tensor = jedi_network->network->getInput(index);
		Dims tensor_dim = tensor->getDimensions();
		for(int index2 = 0; index2 < tensor_dim.nbDims ; index2++) {
			std::cout << tensor_dim.d[index2] << std::endl;
		}
	}*/

	ITensor *tensor = jedi_network->network->getInput(0);
	Dims tensor_dim = tensor->getDimensions();
	//input_dim.channel = tensor_dim.d[1];
	//input_dim.width = tensor_dim.d[2];
	//input_dim.height = tensor_dim.d[3];
	tensor_dim.d[0] = CALIBRATION_BATCH_SIZE;
	tensor_dim.d[1] = 3;
	tensor_dim.d[2] = 640;
	tensor_dim.d[3] = 640;
	input_dim.channel = 3;
	input_dim.width = 640;
	input_dim.height = 640;

	ImageBatchStream *calibrationStream = new ImageBatchStream(tensor_dim, CALIBRATION_BATCH_SIZE, yoloOnnxAppConfig.calib_images_num / CALIBRATION_BATCH_SIZE, yoloOnnxAppConfig.calib_image_path, LOAD_IMAGE_LETTERBOX);
	Int8ImageEntropyCalibrator *calibrator = new Int8ImageEntropyCalibrator(*calibrationStream, 1, calib_table, tensor->getName());

	jedi_network->calibrator = calibrator;
	std::cerr<<"calibration algorithm selected: " << std::to_string((int) jedi_network->calibrator->getAlgorithm()) << std::endl;

	// second input is det_boxes
	ITensor *detBoxTensor = jedi_network->network->getOutput(1); // det_boxes
	Dims detbox_dim = detBoxTensor->getDimensions();
	num_max_detections = detbox_dim.d[1]; // second dimension is maximum number of detections

	return jedi_network;
}


void YoloEnd2EndOnnxApplication::initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	this->network_name = network_name;
	dataset = new ImageDataset(yoloOnnxAppConfig.image_path);
	result_format = new COCOFormat();

	if(yoloOnnxAppConfig.opencv_parallel_num >= 0) {
		cv::setNumThreads(0);
	}
}


void YoloEnd2EndOnnxApplication::initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	for (int i = 0 ; i < thread_number ; i++ ) {
		Detection *dets;
		allocateDetectionBox(maximum_batch_size, num_max_detections, &dets);
		dets_vec.push_back(dets);

		this->detection_num_vec.push_back(std::vector<int>(maximum_batch_size, 0));
	}
}


void YoloEnd2EndOnnxApplication::preprocessing(int thread_id, int input_tensor_index, const char *input_name, int sample_index, int batch_index, IN OUT float *input_buffer)
{
	int image_index = (sample_index + batch_index) % dataset->getSize();
	ImageData *image_data = dataset->getData(image_index);
	int orignal_width = 0;
	int original_height = 0;
	
	loadImageLetterBox((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);
	//loadImageResize((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);
	//loadImageResizeNorm(image_data->path, input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);
	//loadImageLetterBoxNorm((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);

	image_data->width = orignal_width;
	image_data->height = original_height;
}

//,"num_dets" int32 [1,1]
//,"det_boxes" float32[1,100,4]
//,"det_scores" float32[1, 100]
//,"det_classes", int32[1,100]

void YoloEnd2EndOnnxApplication::postprocessing1(int thread_id, int sample_index, IN void **output_buffers, int output_num, int batch)
{
	int *detection_num = static_cast<int*>(output_buffers[0]);
	float *det_boxes = static_cast<float*>(output_buffers[1]);
	float *det_scores = static_cast<float*>(output_buffers[2]);
	int *det_classes = static_cast<int*>(output_buffers[3]);

	for (int batch_index = 0; batch_index < batch; batch_index++) {
		int image_index = (sample_index * batch + batch_index) % dataset->getSize();
		std::list<std::string> detected;

		int32_t num_detections = std::min(num_max_detections, *(detection_num + batch_index));

		//printf("[%d] num_detections: %d\n", sample_index, num_detections);

		ImageData *image_data = dataset->getData(image_index);
		float orig_width = (float) image_data->width;
		float orig_height = (float) image_data->height;
		float scale = 1.0f / std::min(input_dim.height / orig_height, input_dim.width / orig_width);
		char *path = (char *)(image_data->path.c_str());
		int image_id = get_coco_image_id(path);
		int x_offset = (input_dim.width * scale - orig_width) / 2;
  		int y_offset = (input_dim.height * scale - orig_height) / 2;
		
		for (int box_index = 0; box_index < num_detections; box_index++) {
			float *det_box = det_boxes + batch_index*num_max_detections + box_index*4;
			float *det_score = det_scores + batch_index*num_max_detections + box_index;
			int *det_class = det_classes + batch_index*num_max_detections + box_index;

			float xmin = (det_box[0]) * scale - x_offset;
			float ymin = (det_box[1]) * scale - y_offset;
			float xmax = (det_box[2]) * scale - x_offset;
			float ymax = (det_box[3]) * scale - y_offset;
			xmin = std::max(std::min(xmin, (float)(orig_width - 1)), 0.f);
			ymin = std::max(std::min(ymin, (float)(orig_height - 1)), 0.f);
			xmax = std::max(std::min(xmax, (float)(orig_width - 1)), 0.f);
			ymax = std::max(std::min(ymax, (float)(orig_height - 1)), 0.f);

			float bx = xmin;
			float by = ymin;
			float bw = xmax - xmin;
			float bh = ymax - ymin;

			std::stringstream result;
			result<<"{\"image_id\":"<<image_id<<", \"category_id\":"<< result_format->getCOCOIdFromIndex(*det_class) <<", \"bbox\":["<<bx<<", "<<by<<", "<<bw<<", "<<bh<<"], \"score\":"<< *det_score <<"}";
			detected.push_back(result.str());
		}
		result_format->addToDetectedMap(image_index, detected);
	}
}

int YoloEnd2EndOnnxApplication::get_coco_image_id(char *filename) {
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c)
        p = c;
    return atoi(p + 1);
}

void YoloEnd2EndOnnxApplication::postprocessing2(int thread_id, int sample_index, int batch) {
	// do nothing
}


void YoloEnd2EndOnnxApplication::writeResultFile(std::string result_file_name) {
	result_format->writeResultFile(result_file_name);
}

YoloEnd2EndOnnxApplication::~YoloEnd2EndOnnxApplication()
{
	if(this->detection_num_vec.size() > 0) {
		int batch = this->detection_num_vec[0].size();

		while(dets_vec.size() > 0)
		{
			Detection *det = dets_vec.back();
			deallocateDetectionBox(batch * num_max_detections, det);
			dets_vec.pop_back();
		}
	}
	delete dataset;
	delete result_format;
}
