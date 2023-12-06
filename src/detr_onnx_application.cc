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

#include <tkDNN/tkdnn.h>
#include <tkDNN/Int8BatchStream.h>
#include <tkDNN/Int8Calibrator.h>
// #include <tkDNN/Int8MinMaxCalibrator.h>
// #include <tkDNN/Int8HistogramCalibrator.h>
#include "detr_onnx_application.h"

#define NMS 0.45


using namespace nvinfer1;
using namespace nvonnxparser; 

REGISTER_JEDI_APPLICATION(DETROnnxApplication);


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


class OnnxParserLogger2 : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
			std::cout <<"TENSORRT ONNX LOG: "<< msg << std::endl;
    }
} onnx_logger2;


// static inline float logisticActivate(float x){return 1.f/(1.f + expf(-x));}


void DETROnnxApplication::readOnnxFilePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["onnx_file_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		detrOnnxAppConfig.onnx_file_path = data.c_str();

		std::cerr<<"onnx_file_path: "<<detrOnnxAppConfig.onnx_file_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'onnx_file_path' setting in configuration file." << std::endl;
	}
}


void DETROnnxApplication::readCalibImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["calib_image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		detrOnnxAppConfig.calib_image_path = data.c_str();

		std::cerr<<"calib_image_path: "<<detrOnnxAppConfig.calib_image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_image_path' setting in configuration file." << std::endl;
	}
}


void DETROnnxApplication::readCalibImagesNum(libconfig::Setting &setting){
	try {
		const char *data = setting["calib_images_num"];
		detrOnnxAppConfig.calib_images_num = atoi(data);

		std::cerr<<"calib_images_num: "<<detrOnnxAppConfig.calib_images_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_images_num' setting in configuration file." <<std::endl;
	}
}


void DETROnnxApplication::readImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		detrOnnxAppConfig.image_path = data.c_str();

		std::cerr<<"image_path: "<<detrOnnxAppConfig.image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'image_path' setting in configuration file." << std::endl;
	}
}


void DETROnnxApplication::readNamePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["name_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;

		detrOnnxAppConfig.name_path = data.c_str();

		std::cerr<<"name_path: "<<detrOnnxAppConfig.name_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'name_path' setting in configuration file." << std::endl;
	}
}


void DETROnnxApplication::readOpenCVParallelNum(libconfig::Setting &setting) {
	try{	
		const char *data = setting["opencv_parallel_num"];
		detrOnnxAppConfig.opencv_parallel_num = atoi(data);

		std::cerr<<"opencv_parallel_num: "<<detrOnnxAppConfig.opencv_parallel_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'opencv_parallel_num' setting in configuration file. Set -1 as a Default." << std::endl;
		detrOnnxAppConfig.opencv_parallel_num = -1;
	}
}


void DETROnnxApplication::readCustomOptions(libconfig::Setting &setting)
{
	readOnnxFilePath(setting);
	readImagePath(setting);
	readNamePath(setting);
	readOpenCVParallelNum(setting);
	readCalibImagePath(setting);
	readCalibImagesNum(setting);
}


IJediNetwork *DETROnnxApplication::createNetwork(ConfigInstance *basic_config_data)
{
	std::string calib_table = basic_config_data->calib_table;
	TensorRTNetwork *jedi_network = new TensorRTNetwork();

	jedi_network->builder = createInferBuilder(onnx_logger2);

	uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
	jedi_network->network =  jedi_network->builder->createNetworkV2(flag);
	jedi_network->onnx_file_path = detrOnnxAppConfig.onnx_file_path;

	IParser* parser = createParser(*(jedi_network->network), onnx_logger2);

	// TODO: onnx file path
	parser->parseFromFile(detrOnnxAppConfig.onnx_file_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
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

	ITensor *tensor = jedi_network->network->getInput(0);
	Dims tensor_dim = tensor->getDimensions();
	input_dim.channel = tensor_dim.d[1];
	input_dim.width = tensor_dim.d[2];
	input_dim.height = tensor_dim.d[3];
	dataDim_t dim(tensor_dim.d[0], tensor_dim.d[1], tensor_dim.d[2], tensor_dim.d[3]);
	BatchStream *calibrationStream = new BatchStream(dim, 1, detrOnnxAppConfig.calib_images_num, detrOnnxAppConfig.calib_image_path);
	Int8EntropyCalibrator *calibrator = new Int8EntropyCalibrator(*calibrationStream, 1, calib_table, "data");
	// Int8MinMaxCalibrator *calibrator = new Int8MinMaxCalibrator(*calibrationStream, 1, calib_table , "data");
	// Int8HistogramCalibrator *calibrator = new Int8HistogramCalibrator(*calibrationStream, 1, calib_table , "data");
	jedi_network->calibrator = calibrator;
	std::cerr<<"calibration algorithm selected: " << std::to_string((int) jedi_network->calibrator->getAlgorithm()) << std::endl;

	// object detection has two outputs
	ITensor *logitsTensor = jedi_network->network->getOutput(0);
	Dims logit_dim = logitsTensor->getDimensions();
	num_detections = logit_dim.d[1];
	num_classes = logit_dim.d[2];

	return jedi_network;
}


void DETROnnxApplication::initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	this->network_name = network_name;
	dataset = new ImageDataset(detrOnnxAppConfig.image_path);
	result_format = new COCOFormat();

	if(detrOnnxAppConfig.opencv_parallel_num >= 0) {
		cv::setNumThreads(0);
	}
}


void DETROnnxApplication::initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	for (int i = 0 ; i < thread_number ; i++ ) {
		Detection *dets;
		allocateDetectionBox(maximum_batch_size, &dets);
		dets_vec.push_back(dets);

		this->detection_num_vec.push_back(std::vector<int>(maximum_batch_size, 0));
	}
}


void DETROnnxApplication::preprocessing(int thread_id, int input_tensor_index, const char *input_name, int sample_index, int batch_index, IN OUT float *input_buffer)
{
	int image_index = (sample_index + batch_index) % dataset->getSize();
	ImageData *image_data = dataset->getData(image_index);
	int orignal_width = 0;
	int original_height = 0;

	loadImageResizeNorm(image_data->path, input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);
	//loadImageLetterBoxNorm((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);

	image_data->width = orignal_width;
	image_data->height = original_height;
}


void DETROnnxApplication::postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch)
{
	for (int batch_index = 0; batch_index < batch; batch_index++) {
		int image_index = (sample_index * batch + batch_index) % dataset->getSize();
		std::list<std::string> detected;

		for (int box_index = 0; box_index < num_detections; box_index++) {
			float* logit = output_buffers[0] + batch_index*num_detections*num_classes + box_index*num_classes;
			float confidence = 0.0;
			int label = 0;

			// //print 10 items in logit	
			// for (int i = 0; i < 10; i++) {
			// 	std::cout << logit[i] << " " << std::flush;
			// }
			// std::cout << std::endl;

			computeConfidenceAndLabels(logit, confidence, label);

			//if (confidence > 0) {
			if (confidence > PRINT_THRESH) {
				ImageData *image_data = dataset->getData(image_index);
				float orig_width = (float) image_data->width;
				float orig_height = (float) image_data->height;

				char *path = (char *)(image_data->path.c_str());
				int image_id = get_coco_image_id(path);

				// float* logit = output_buffers[0] + batch_index + num_detections*num_classes + box_index * num_classes;
				float *pred_box = output_buffers[1] + batch_index*num_detections*4 + box_index * 4;

				//rescale box 
				// float rescale_width = orig_width / this->input_dim.width;
				// float rescale_height = orig_height / this->input_dim.height;

				float bx = (pred_box[0] - pred_box[2]/2.) * orig_width;
				float by = (pred_box[1] - pred_box[3]/2.) * orig_height;
				float bw = pred_box[2] * orig_width;
				float bh = pred_box[3] * orig_height;

				bx = bx < 0 ? 0 : bx;
				by = by < 0 ? 0 : by;
				bw = bw > orig_width ? orig_width : bw;
				bh = bh > orig_height ? orig_height : bh;

				// std::cout << "image_id: " << image_id << ", category_id: " << label << ", bbox: [" << bx << ", " << by << ", " << bw << ", " << bh << "], score: " << confidence << std::endl;

				std::stringstream result;
				result<<"{\"image_id\":"<<image_id<<", \"category_id\":"<<label<<", \"bbox\":["<<bx<<", "<<by<<", "<<bw<<", "<<bh<<"], \"score\":"<<confidence<<"}";
				detected.push_back(result.str());
			}
		}
		result_format->addToDetectedMap(image_index, detected);
	}
}

int DETROnnxApplication::get_coco_image_id(char *filename) {
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c)
        p = c;
    return atoi(p + 1);
}

void DETROnnxApplication::postprocessing2(int thread_id, int sample_index, int batch) {
	// do nothing
}


#define DETR_MAX_CLASSES (91)

void DETROnnxApplication::computeConfidenceAndLabels(float *logit, float &confidence, int &label)
{
	// softmax(logit) => prob
	softmax(logit); 

	int max_classes = std::min(num_classes, DETR_MAX_CLASSES);
	
	// get max score and class
	float max_value = 0.0;
	int max_index = 0;
	for(int class_index = 0; class_index < max_classes; class_index++) {
		if (logit[class_index] > max_value) {
			max_value = logit[class_index];
			max_index = class_index;
		}
	} 
	confidence = max_value;
	label = max_index;
}


void DETROnnxApplication::softmax(float *logit){
	// int iter1 = 0;
	// float sum = 0;
	// for (iter1 = 0; iter1 < num_classes; iter1++) {
	// 	logit[iter1] = expf(logit[iter1]);
	// 	sum += logit[iter1];
	// }

	// for (iter1 = 0; iter1 < num_classes; iter1++) {
	// 	logit[iter1] /= sum;
	// }
	int i = 0;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < num_classes; ++i){
        if(logit[i] > largest) largest = logit[i];
    }

    for(i = 0; i < num_classes; ++i){
        float e = exp(logit[i] - largest);
        sum += e;
        logit[i] = e;
    }

    for(i = 0; i < num_classes; ++i){
        logit[i] /= sum;
    }
}

void DETROnnxApplication::writeResultFile(std::string result_file_name) {
	result_format->writeResultFile(result_file_name);
}

DETROnnxApplication::~DETROnnxApplication()
{
	int batch = this->detection_num_vec[0].size();

	while(dets_vec.size() > 0)
	{
		Detection *det = dets_vec.back();
		deallocateDetectionBox(batch * NBOXES, det);
		dets_vec.pop_back();
	}
	delete dataset;
	delete result_format;
}
