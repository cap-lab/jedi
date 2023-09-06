#include <libconfig.h++>
#include <cstring>
#include <fstream>
#include <sstream>
#include <math.h>
#include <limits>

#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "image_opencv.h"
#include "box.h"
#include "region_wrapper.h"
#include "tensorrt_network.h"
#include "image_cls_onnx_application.h"

#include <tkDNN/tkdnn.h>
#include "int8_image_batch_stream.h"
#include "int8_calibrator.h"


using namespace nvinfer1;
using namespace nvonnxparser; 

REGISTER_JEDI_APPLICATION(ImageClsOnnxApplication);


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

class OnnxParserLogger3 : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        //if (severity <= Severity::kWARNING)
		std::cout <<"TENSORRT ONNX LOG: "<< msg << std::endl;
    }
} onnx_logger3;

void ImageClsOnnxApplication::readOnnxFilePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["onnx_file_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		imageClsOnnxAppConfig.onnx_file_path = data.c_str();

		std::cerr<<"onnx_file_path: "<<imageClsOnnxAppConfig.onnx_file_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'onnx_file_path' setting in configuration file." << std::endl;
	}
}

void ImageClsOnnxApplication::readCalibImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["calib_image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		imageClsOnnxAppConfig.calib_image_path = data.c_str();

		std::cerr<<"calib_image_path: "<<imageClsOnnxAppConfig.calib_image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_image_path' setting in configuration file." << std::endl;
	}
}

void ImageClsOnnxApplication::readCalibImagesNum(libconfig::Setting &setting){
	try {
		const char *data = setting["calib_images_num"];
		imageClsOnnxAppConfig.calib_images_num = atoi(data);

		std::cerr<<"calib_images_num: "<<imageClsOnnxAppConfig.calib_images_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_images_num' setting in configuration file." <<std::endl;
	}
}


void ImageClsOnnxApplication::readImagePreprocessingOption(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["image_preprocessing"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;

		if (data == "resize") {
			imageClsOnnxAppConfig.preprocessing_option = LOAD_IMAGE_RESIZE;
		}
		else if (data == "letterbox") {
			imageClsOnnxAppConfig.preprocessing_option = LOAD_IMAGE_LETTERBOX;

		}
		else if(data == "resize_norm") {
			imageClsOnnxAppConfig.preprocessing_option = LOAD_IMAGE_RESIZE_NORM;
		}
		else if(data == "resize_crop_norm") {
			imageClsOnnxAppConfig.preprocessing_option = LOAD_IMAGE_RESIZE_CROP_NORM;
		}
		else if (data == "resize_crop") {
			imageClsOnnxAppConfig.preprocessing_option = LOAD_IMAGE_RESIZE_CROP;
		}
		else if (data == "resize_crop_norm_ml") {
			imageClsOnnxAppConfig.preprocessing_option = LOAD_IMAGE_RESIZE_CROP_NORM_ML;
		}
		else {
			imageClsOnnxAppConfig.preprocessing_option = LOAD_IMAGE_RESIZE;
		}

		std::cerr<<"image_preprocessing: "<< data <<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_image_path' setting in configuration file. Set resize as a default." << std::endl;
		imageClsOnnxAppConfig.preprocessing_option = LOAD_IMAGE_RESIZE;
	}
}

void ImageClsOnnxApplication::readImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		imageClsOnnxAppConfig.image_path = data.c_str();

		std::cerr<<"image_path: "<<imageClsOnnxAppConfig.image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'image_path' setting in configuration file." << std::endl;
	}
}

void ImageClsOnnxApplication::readLabelPath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["label_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;

		imageClsOnnxAppConfig.label_path = data.c_str();

		std::cerr<<"label_path: "<<imageClsOnnxAppConfig.label_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'label_path' setting in configuration file." << std::endl;
	}
}

void ImageClsOnnxApplication::readOpenCVParallelNum(libconfig::Setting &setting) {
	try{	
		const char *data = setting["opencv_parallel_num"];
		imageClsOnnxAppConfig.opencv_parallel_num = atoi(data);

		std::cerr<<"opencv_parallel_num: "<<imageClsOnnxAppConfig.opencv_parallel_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'opencv_parallel_num' setting in configuration file. Set -1 as a Default." << std::endl;
		imageClsOnnxAppConfig.opencv_parallel_num = -1;
	}
}



void ImageClsOnnxApplication::readCustomOptions(libconfig::Setting &setting)
{
	readOnnxFilePath(setting);
	readImagePath(setting);
	readLabelPath(setting);
	readOpenCVParallelNum(setting);
	readCalibImagePath(setting);
	readCalibImagesNum(setting);
	readImagePreprocessingOption(setting);
}

IJediNetwork *ImageClsOnnxApplication::createNetwork(ConfigInstance *basic_config_data)
{
	std::string calib_table = basic_config_data->calib_table;
	TensorRTNetwork *jedi_network = new TensorRTNetwork();

	jedi_network->builder = createInferBuilder(onnx_logger3);

	uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
	jedi_network->network =  jedi_network->builder->createNetworkV2(flag);
	jedi_network->onnx_file_path = imageClsOnnxAppConfig.onnx_file_path;

	IParser* parser = createParser(*(jedi_network->network), onnx_logger3);

	// TODO: onnx file path
	parser->parseFromFile(imageClsOnnxAppConfig.onnx_file_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
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
	// dataDim_t dim(tensor_dim.d[0],tensor_dim.d[1], tensor_dim.d[2], tensor_dim.d[3]);
	//dataDim_t dim(basic_config_data->batch,tensor_dim.d[1], tensor_dim.d[2], tensor_dim.d[3]);
	ImageBatchStream *calibrationStream = new ImageBatchStream(tensor_dim, 1, imageClsOnnxAppConfig.calib_images_num, imageClsOnnxAppConfig.calib_image_path, imageClsOnnxAppConfig.preprocessing_option);
	Int8ImageEntropyCalibrator *calibrator = new Int8ImageEntropyCalibrator(*calibrationStream, 1, calib_table , tensor->getName());
	jedi_network->calibrator = calibrator;
	std::cerr<<"calibration algorithm selected: " << std::to_string((int) jedi_network->calibrator->getAlgorithm()) << std::endl;

	ITensor *tensor_output = jedi_network->network->getOutput(0);
	tensor_dim = tensor_output->getDimensions();
	class_num = tensor_dim.d[1];

	return jedi_network;
}

void ImageClsOnnxApplication::initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	this->network_name = network_name;
	dataset = new ImageDataset(imageClsOnnxAppConfig.image_path);
	result_format = new ImagenetFormat();
	//class_num = result_format->class_num;

	if(imageClsOnnxAppConfig.opencv_parallel_num >= 0) {
		cv::setNumThreads(0);
	}
}

void ImageClsOnnxApplication::preprocessing(int thread_id, int sample_index, int batch_index, IN OUT float *input_buffer)
{
	int image_index = (sample_index + batch_index) % dataset->getSize();

	ImageData *image_data = dataset->getData(image_index);
	int original_width = 0;
	int original_height = 0;

	switch(imageClsOnnxAppConfig.preprocessing_option) {
		case LOAD_IMAGE_RESIZE:
			loadImageResize((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &original_width, &original_height, input_buffer);
			break;
		case LOAD_IMAGE_LETTERBOX:
			loadImageLetterBox((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &original_width, &original_height, input_buffer);
		case LOAD_IMAGE_RESIZE_NORM:
			loadImageResizeNorm((char *)image_data->path.c_str(), input_dim.width, input_dim.height, input_dim.channel, &original_width, &original_height, input_buffer);
			break;
		case LOAD_IMAGE_RESIZE_CROP_NORM:
			loadImageResizeCropNorm((char *)(image_data->path.c_str()), input_dim.width + 32, input_dim.height + 32, input_dim.channel, input_dim.width, input_buffer); // efficient former
			break;
		case LOAD_IMAGE_RESIZE_CROP:
			loadImageResizeCrop((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, input_buffer); // efficient net
			break;
		case LOAD_IMAGE_RESIZE_CROP_NORM_ML:
			loadImageResizeCropNormML((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, input_buffer); // resnet mlperf
			break;
		default:
			loadImageResize((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &original_width, &original_height, input_buffer);
			break;
	}

	image_data->width = original_width;
	image_data->height = original_height;
}

void ImageClsOnnxApplication::initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	std::string label_path = imageClsOnnxAppConfig.label_path;
	std::ifstream fp(label_path);
	std::string line = "0000000";

	if(class_num > result_format->class_num) {
		labels.push_back(line);
	}

	if(fp.is_open()) {
		while(std::getline(fp, line)) {
			labels.push_back(line);	
		}
		fp.close();
	}
}

char* ImageClsOnnxApplication::nolibStrStr(const char *s1, const char *s2) {
	//fprintf(stderr, "s1: |%s|, s2: |%s|\n", s1, s2);
	int i;
	if (*s2 == '\0') {
		return (char *)s1;
	} else {
		for (; *s1; s1 ++) {
			if (*s1 == *s2) {
				for (i = 1; *(s1 + i) == *(s2 + i); i++);
				if (i == strlen(s2))
					return (char *)s1;
			}
		}
		return nullptr;
	}
}

int ImageClsOnnxApplication::generateTruths(std::string path) {
	for (size_t iter = 0; iter < labels.size(); iter++) {
		std::string label = labels[iter];
		if (nolibStrStr(path.c_str(), label.c_str())) {
			return (int) iter;
		}
	}
	std::cerr<<"an answer of "<<path<<" is not found: "<<std::endl;
	return -1;
}

void ImageClsOnnxApplication::softmax(float *logit){
	int i = 0;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < class_num; ++i){
        if(logit[i] > largest) largest = logit[i];
    }

    for(i = 0; i < class_num; ++i){
        float e = exp(logit[i] - largest);
        sum += e;
        logit[i] = e;
    }

    for(i = 0; i < class_num; ++i){
        logit[i] /= sum;
    }
}



void ImageClsOnnxApplication::postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch)
{
	float *data_to_check = output_buffers[0];

	for (int iter1 = 0; iter1 < batch; iter1++) {
		int image_index = (sample_index * batch + iter1) % dataset->getSize();
		ImageData *data = dataset->getData(image_index);

		std::string path = data->path;

		int answer = generateTruths(path);	
		//if(labels.size() != class_num) { // class_num value is 1001 and labels.size() is 1000, increase the value of answer
		//	answer = answer + 1;
		//}
		//softmax(&(data_to_check[iter1*class_num]));

		int guess = -1;
		float max_value = std::numeric_limits<float>::lowest();
		//float max_value = std::numeric_limits<float>::min();
		for (int iter2 = 0; iter2 < class_num; iter2++) {
			if (max_value < data_to_check[iter1 * class_num + iter2]) {
				max_value = data_to_check[iter1 * class_num + iter2];
				guess = iter2;
			}
		}

		result_format->recordIsCorrect((answer == guess));
	}
}

void ImageClsOnnxApplication::postprocessing2(int thread_id, int sample_index, int batch) {
	// do nothing
}

void ImageClsOnnxApplication::writeResultFile(std::string result_file_name) {
	result_format->writeResultFile(result_file_name);
}

ImageClsOnnxApplication::~ImageClsOnnxApplication()
{
	labels.clear();
	delete dataset;
	delete result_format;
}
