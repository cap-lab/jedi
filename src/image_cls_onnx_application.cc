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
#include <tkDNN/Int8BatchStream.h>
#include <tkDNN/Int8Calibrator.h>


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
	dataDim_t dim(basic_config_data->batch,tensor_dim.d[1], tensor_dim.d[2], tensor_dim.d[3]);
	BatchStream *calibrationStream = new BatchStream(dim, 1, imageClsOnnxAppConfig.calib_images_num, imageClsOnnxAppConfig.calib_image_path);
	Int8EntropyCalibrator *calibrator = new Int8EntropyCalibrator(*calibrationStream, 1, calib_table , "data");
	jedi_network->calibrator = calibrator;
	std::cerr<<"calibration algorithm selected: " << std::to_string((int) jedi_network->calibrator->getAlgorithm()) << std::endl;

	return jedi_network;
}

void ImageClsOnnxApplication::initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	this->network_name = network_name;
	dataset = new ImageDataset(imageClsOnnxAppConfig.image_path);
	result_format = new ImagenetFormat();
	class_num = result_format->class_num;

	if(imageClsOnnxAppConfig.opencv_parallel_num >= 0) {
		cv::setNumThreads(0);
	}
}

void ImageClsOnnxApplication::preprocessing(int thread_id, int sample_index, int batch_index, IN OUT float *input_buffer)
{
	int image_index = (sample_index + batch_index) % dataset->getSize();

	ImageData *image_data = dataset->getData(image_index);
	int orignal_width = 0;
	int original_height = 0;

	// std::cerr<<"width: "<<input_dim.width<<" height: "<<input_dim.height<<std::endl;
	// loadImageResize((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel, &orignal_width, &original_height, input_buffer);
	loadImageResizeCropNorm((char *)(image_data->path.c_str()), 256, 256, 3, 224, input_buffer);

	image_data->width = orignal_width;
	image_data->height = original_height;
}

void ImageClsOnnxApplication::initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	std::string label_path = imageClsOnnxAppConfig.label_path;
	std::ifstream fp(label_path);
	std::string line;

	if(fp.is_open()) {
		while(std::getline(fp, line)) {
			labels.push_back(line);	
		}
		fp.close();
	}
}

char* ImageClsOnnxApplication::nolibStrStr(const char *s1, const char *s2) {
	// fprintf(stderr, "s1: |%s|, s2: |%s|\n", s1, s2);
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
	for (int iter = 0; iter < class_num; iter++) {
		std::string label = labels[iter];
		if (nolibStrStr(path.c_str(), label.c_str())) {
			return iter;
		}
	}
	std::cerr<<"an answer of "<<path<<" is not found: "<<std::endl;
	return -1;
}

void ImageClsOnnxApplication::postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch)
{
	float *data_to_check = output_buffers[0];

	for (int iter1 = 0; iter1 < batch; iter1++) {
		int image_index = (sample_index * batch + iter1) % dataset->getSize();
		ImageData *data = dataset->getData(image_index);

		std::string path = data->path;

		int answer = generateTruths(path);	

		int guess = -1;
		float max_value = std::numeric_limits<float>::min();
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
