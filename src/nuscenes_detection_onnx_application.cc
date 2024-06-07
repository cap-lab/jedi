#include <libconfig.h++>
#include <cstring>
#include <fstream>
#include <sstream>
#include <math.h>
#include <limits>

#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "tensorrt_network.h"
#include "nuscenes_detection_onnx_application.h"

#include <tkDNN/tkdnn.h>


using namespace nvinfer1;
using namespace nvonnxparser; 

REGISTER_JEDI_APPLICATION(NuscenesDetectionOnnxApplication);

#define X_STEP 0.2f
#define Y_STEP 0.2f
#define X_MIN -51.2f
#define X_MAX 51.2f
#define Y_MIN -51.2f
#define Y_MAX 51.2f
#define Z_MIN -5.0f
#define Z_MAX 3.0f
#define PI 3.141592653f

// paramerters for preprocess
#define BEV_W 512
#define BEV_H 512
#define MAX_PILLARS 30000
#define MAX_POINT_IN_PILLARS 20
#define FEATURE_NUM 10
#define THREAD_NUM 1

// paramerters for postprocess
#define SCORE_THRESHOLD 0.1f
#define NMS_THREAHOLD 0.2f
#define INPUT_NMS_MAX_SIZE 1000
#define OUT_SIZE_FACTOR 4.0f
#define TASK_NUM 6
#define REG_CHANNEL 2
#define HEIGHT_CHANNEL 1
#define ROT_CHANNEL 2
#define VEL_CHANNEL 2
#define DIM_CHANNEL 3
#define OUTPUT_H 128
#define OUTPUT_W 128

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

class OnnxParserLogger4 : public ILogger{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        //if (severity <= Severity::kWARNING)
		std::cout <<"TENSORRT ONNX LOG: "<< msg << std::endl;
    }
} onnx_logger4;

void NuscenesDetectionOnnxApplication::readOnnxFilePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["onnx_file_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		nuscenesOnnxAppConfig.onnx_file_path = data.c_str();

		std::cerr<<"onnx_file_path: "<<nuscenesOnnxAppConfig.onnx_file_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'onnx_file_path' setting in configuration file." << std::endl;
		exit(EXIT_FAILURE);
	}
}

void NuscenesDetectionOnnxApplication::readOptimizationProfileFilePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["optimization_cfg_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		nuscenesOnnxAppConfig.optimization_cfg_path = data.c_str();
		std::cerr<<"optimization_cfg_path: "<<nuscenesOnnxAppConfig.optimization_cfg_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'optimization_cfg_path' setting in configuration file." << std::endl;
		//exit(EXIT_FAILURE);
	}
}

void NuscenesDetectionOnnxApplication::readLidarListPath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["lidar_list_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		nuscenesOnnxAppConfig.lidar_list_path = data.c_str();
		std::cerr<<"lidar_list_path: "<<nuscenesOnnxAppConfig.lidar_list_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'lidar_list_path' setting in configuration file." << std::endl;
		exit(EXIT_FAILURE);
	}
}


void NuscenesDetectionOnnxApplication::readCustomOptions(libconfig::Setting &setting)
{
	readOnnxFilePath(setting);
	readOptimizationProfileFilePath(setting);
	readLidarListPath(setting);
}

IJediNetwork *NuscenesDetectionOnnxApplication::createNetwork(ConfigInstance *basic_config_data)
{
	std::string calib_table = basic_config_data->calib_table;
	TensorRTNetwork *jedi_network = new TensorRTNetwork();

	jedi_network->builder = createInferBuilder(onnx_logger4);

	uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
	jedi_network->network =  jedi_network->builder->createNetworkV2(flag);
	jedi_network->onnx_file_path = nuscenesOnnxAppConfig.onnx_file_path;

	IParser* parser = createParser(*(jedi_network->network), onnx_logger4);

	// TODO: onnx file path
	parser->parseFromFile(nuscenesOnnxAppConfig.onnx_file_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
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
	//libconfig::Config cfg;
    //readOptimizationConfigFile(&cfg, nuscenesOnnxAppConfig.optimization_cfg_path );
	//libconfig::Setting &setting = cfg.lookup("configs");
	jedi_network->optimization_cfg_path = nuscenesOnnxAppConfig.optimization_cfg_path;

	//ITensor *tensor = jedi_network->network->getInput(0);
	//Dims tensor_dim = tensor->getDimensions();
	//input_dim.channel = tensor_dim.d[1];
	//input_dim.width = tensor_dim.d[2];
	//input_dim.height = tensor_dim.d[3];

    // int num_inputs = jedi_network->network->getNbInputs();
	// for (int i = 0 ; i < num_inputs ; i++) {
	// 	ITensor *tensor_input = jedi_network->network->getInput(i);
    //     std::cout << "input name: " << tensor_input->getName() << ", index: " << i << std::endl;
	// }

	// int num_outputs = jedi_network->network->getNbOutputs();
	// for (int i = 0 ; i < num_outputs ; i++) {
	// 	ITensor *tensor_output = jedi_network->network->getOutput(i);
	// 	outputIndexMap[tensor_output->getName()] = i;
    //     std::cout << "output name: " << tensor_output->getName() << ", index: " << i << std::endl;
	// }

	//ITensor *tensor_output = jedi_network->network->getOutput(0);
	//tensor_dim = tensor_output->getDimensions();
	//class_num = tensor_dim.d[1];

	return jedi_network;
}

void NuscenesDetectionOnnxApplication::initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	this->network_name = network_name;
	dataset = new LidarDataset(nuscenesOnnxAppConfig.lidar_list_path);
    result_format = new NuscenesFormat();
	//class_num = result_format->class_num;

	for(int i = 0 ; i < thread_number ; i++) {
		float *feature = nullptr;
		int *indices = nullptr;
		inputBuffers.emplace_back(nullptr);
		inputBufferSizes.emplace_back(0);

		feature = (float *) malloc(MAX_PILLARS*FEATURE_NUM*MAX_POINT_IN_PILLARS*sizeof(float));
		if(feature == nullptr) {
			std::cerr << "[Error] Malloc Feature Memory Failed! Size: " << MAX_PILLARS*FEATURE_NUM*MAX_POINT_IN_PILLARS*sizeof(float) << std::endl;
			exit(EXIT_FAILURE);
		}

		indices = (int *) malloc(MAX_PILLARS*2*sizeof(int));
		if(indices == nullptr) {
			std::cerr << "[Error] Malloc Indices Memory Failed! Size: " << MAX_PILLARS*2*sizeof(int) << std::endl;
			exit(EXIT_FAILURE);
		}

		featureList.emplace_back(feature);
		indicesList.emplace_back(indices);
        current_lidar_indexs.emplace_back(-1);
	}
}


bool NuscenesDetectionOnnxApplication::readBinFile(int thread_id, std::string& filename, float*& bufPtr, int& pointNum)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
		std::cerr << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

	if (inputBufferSizes[thread_id] < fileSize) {
		if(inputBuffers[thread_id] != nullptr) {
			free(inputBuffers[thread_id]);
		}
		bufPtr = (float *) malloc(fileSize);
		if(bufPtr == nullptr){
			std::cerr << "[Error] Malloc Memory Failed! Size: " << fileSize << std::endl;
			return false;
		}

		inputBuffers[thread_id] = bufPtr;
		inputBufferSizes[thread_id] = fileSize;
	} else {
		bufPtr = inputBuffers[thread_id];
	}

    // read the data:
    file.read((char*) bufPtr, fileSize);
    file.close();
    
    constexpr int featureNum = 5;
    pointNum = fileSize /sizeof(float) / featureNum;
    if( fileSize /sizeof(float) % featureNum != 0){
		std::cerr << "[Error] File Size Error! " << fileSize << std::endl;
    }
	//std::cout << "[INFO] pointNum : " << pointNum << ", filename: " << filename << std::endl;
    return true;
}

static void debugWrite(void *data, int sizeToWrite, const char *output_file_path) {
    FILE *f = fopen(output_file_path, "wb");
    fwrite(data, 1, sizeToWrite, f);
    fclose(f);
}


void NuscenesDetectionOnnxApplication::preprocessing(int thread_id, int input_tensor_index, const char *input_name, int sample_index, int batch_index, IN OUT float *input_buffer)
{
	// preprocessing logic for the single sample
	int lidar_index = (sample_index + batch_index) % dataset->getSize();
	if (current_lidar_indexs[thread_id] != lidar_index) {
		float *point_ptr;
		int point_num = 0;
		bool readBinOk = false;

		memset(indicesList[thread_id], -1, MAX_PILLARS*2*sizeof(int));
		memset(featureList[thread_id], 0, MAX_PILLARS*FEATURE_NUM*MAX_POINT_IN_PILLARS*sizeof(float));

		readBinOk = readBinFile(thread_id, dataset->getData(lidar_index)->path, point_ptr, point_num);
		if(readBinOk == false) {
			exit(EXIT_FAILURE);
		}

		makePillars(point_ptr, featureList[thread_id], indicesList[thread_id], point_num, 0, MAX_PILLARS/THREAD_NUM);

		current_lidar_indexs[thread_id] = lidar_index;

		memcpy(input_buffer, featureList[thread_id], MAX_PILLARS*FEATURE_NUM*MAX_POINT_IN_PILLARS*sizeof(float));

        //debugWrite((void *) input_buffer, MAX_PILLARS*FEATURE_NUM*MAX_POINT_IN_PILLARS*sizeof(float), "feature_backup.binary");
	} else {
		memcpy(input_buffer, indicesList[thread_id], MAX_PILLARS*2*sizeof(int));
        //debugWrite((void *) input_buffer, MAX_PILLARS*2*sizeof(int), "indices_backup.binary");
	}
}

void NuscenesDetectionOnnxApplication::makePillars(float* points, float* feature, int* indices, int pointNum, int threadIdx, int pillarsPerThread){
    // 0 ~ MAX_POINT_IN_PILLARS
    unsigned short pointCount[MAX_PILLARS] = {0};

    // 0 ~ MAX_PILLARS
    int pillarsIndices[BEV_W*BEV_H] = {0};
    int pillarCount = threadIdx*pillarsPerThread;

    memset(pillarsIndices, -1, BEV_W*BEV_H*sizeof(int));

    for(int idx = 0; idx < pointNum; idx++){
        
        auto x = points[idx*5];
        auto y = points[idx*5+1];
        auto z = points[idx*5+2];
        if(x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX || 
           z < Z_MIN || z > Z_MAX)
           continue;

        int xIdx = int((x-X_MIN)/X_STEP);
        int yIdx = int((y-Y_MIN)/Y_STEP);
        
        if(xIdx % THREAD_NUM != threadIdx)
            continue;

        int pillarIdx = yIdx*BEV_W+xIdx;

        if(pillarIdx >= BEV_W*BEV_H || pillarIdx < 0) {
            std::cout << "xIdx: " << pillarIdx << ", yIdx: " << yIdx << std::endl;
            std::cout << "pillarsIndices[" << pillarIdx << "]: " << pillarsIndices[pillarIdx] << std::endl;
        }

        auto pillarCountIdx = pillarsIndices[pillarIdx];

        if (pillarCountIdx >= MAX_PILLARS || (pillarCountIdx == -1 && pillarCount*2 + 1 >= MAX_PILLARS*2)) {
            continue;
        }

        // new pillar index
        if(pillarCountIdx == -1){
            pillarCountIdx = pillarCount;
            pillarsIndices[pillarIdx] = pillarCount;
            indices[pillarCount*2 + 1] = pillarIdx;
            ++pillarCount;
        }


        // if(pillarCountIdx < 0 || pillarCountIdx >= MAX_PILLARS)  {
        //     std::cout << "pointCount[" << pillarCountIdx << "]: " << pointCount[pillarCountIdx] << std::endl;
        // }

        auto pointNumInPillar = pointCount[pillarCountIdx];
        if(pointNumInPillar > MAX_POINT_IN_PILLARS - 1)
            continue;


        //std::cout << "pillarsIndices2[" << pillarIdx << "]: " << pillarsIndices[pillarIdx] << std::endl;
        //std::cout << "pointCount2[" << pillarCountIdx << "]: " << pointCount[pillarCountIdx] << std::endl;

        feature[                                     pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = x;
        feature[1*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = y;
        feature[2*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = z; // z
        feature[3*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = points[idx*5+3]; // instence
        feature[4*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = points[idx*5+4]; // time_lag

        feature[8*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = x - (xIdx*X_STEP + X_MIN + X_STEP/2);
        feature[9*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = y - (yIdx*Y_STEP + Y_MIN + Y_STEP/2);

        ++pointNumInPillar;
        pointCount[pillarCountIdx] = pointNumInPillar;
        
    }
    
    for(int pillarIdx = threadIdx*pillarsPerThread; pillarIdx < (threadIdx+1)*pillarsPerThread; pillarIdx++)
    {
        float xCenter = 0;
        float yCenter = 0;
        float zCenter = 0;
        auto pointNum = pointCount[pillarIdx];
        for(int pointIdx=0; pointIdx < pointNum; pointIdx++)
        {
            auto x = feature[                                     pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            auto y = feature[1*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            auto z = feature[2*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            xCenter += x;
            yCenter += y;
            zCenter += z;
        }
        xCenter = xCenter / pointNum;
        yCenter = yCenter / pointNum;
        zCenter = zCenter / pointNum;
        
        for(int pointIdx=0; pointIdx < pointNum; pointIdx++)
        {    
            auto x = feature[                                     pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            auto y = feature[1*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            auto z = feature[2*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
       
            feature[5*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx] = x - xCenter;
            feature[6*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx] = y - yCenter;
            feature[7*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx] = z - zCenter;

        }
    }
    
}

void NuscenesDetectionOnnxApplication::initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
    std::vector<std::string> outputName{ "594","598","606","610","618","622","630","634","642","646",
                                         "654","658","666","670","678","682","690","694","702","706",
                                         "714","718","726","730","736","737","738","740","741","742",
                                         "744","745","746","748","749","750","752","753","754","756",
                                         "757","758"};

	for (int i = 0 ; i < outputName.size() ; i++) {
		outputIndexMap[outputName[i]] = i;
	}
}


inline void RotateAroundCenter(Box& box, float (&corner)[4][2], float& cosVal, float& sinVal, float (&cornerANew)[4][2]){
    
    for(auto idx = 0; idx < 4; idx++){
        auto x = corner[idx][0];
        auto y = corner[idx][1];

        cornerANew[idx][0] = (x - box.x) * cosVal + (y - box.y) * (-sinVal) + box.x;
        cornerANew[idx][1] = (x - box.x) * sinVal + (y - box.y) * cosVal + box.y;
    }
}
inline void FindMaxMin(float (&box)[4][2], float& maxVAl, float& minVAl, int xyIdx){
    
    maxVAl = box[0][xyIdx];
    minVAl = box[0][xyIdx];
    
    for(auto idx=0; idx < 4; idx++){
        if (maxVAl < box[idx][xyIdx])
            maxVAl = box[idx][xyIdx];

        if (minVAl > box[idx][xyIdx])
            minVAl = box[idx][xyIdx];

    }
}

inline void AlignBox(float (&cornerRot)[4][2], float (&cornerAlign)[2][2]){

    float maxX = 0;
    float minX = 0;
    float maxY = 0;
    float minY = 0;

    FindMaxMin(cornerRot, maxX, minX, 0); // 0 mean X
    FindMaxMin(cornerRot, maxY, minY, 1); // 1 mean X

    cornerAlign[0][0] = minX;
    cornerAlign[0][1] = minY;
    cornerAlign[1][0] = maxX;
    cornerAlign[1][1] = maxY;

}

inline float IoUBev(Box& boxA, Box& boxB){
   
    float ax1 = boxA.x - boxA.l/2;
    float ax2 = boxA.x + boxA.l/2;
    float ay1 = boxA.y - boxA.w/2;
    float ay2 = boxA.y + boxA.w/2;

    float bx1 = boxB.x - boxB.l/2;
    float bx2 = boxB.x + boxB.l/2;
    float by1 = boxB.y - boxB.w/2;
    float by2 = boxB.y + boxB.w/2;

    float cornerA[4][2] = {{ax1, ay1}, {ax1, ay2},
                         {ax2, ay1}, {ax2, ay2}};
    float cornerB[4][2] = {{bx1, ay1}, {bx1, by2},
                         {bx2, by1}, {bx2, by2}};
    
    float cornerARot[4][2] = {0};
    float cornerBRot[4][2] = {0};

    float cosA = cos(boxA.theta), sinA = sin(boxA.theta);
    float cosB = cos(boxB.theta), sinB = sin(boxB.theta);

    RotateAroundCenter(boxA, cornerA, cosA, sinA, cornerARot);
    RotateAroundCenter(boxB, cornerB, cosB, sinB, cornerBRot);

    float cornerAlignA[2][2] = {0};
    float cornerAlignB[2][2] = {0};

    AlignBox(cornerARot, cornerAlignA);
    AlignBox(cornerBRot, cornerAlignB);
    
    float sBoxA = (cornerAlignA[1][0] - cornerAlignA[0][0]) * (cornerAlignA[1][1] - cornerAlignA[0][1]);
    float sBoxB = (cornerAlignB[1][0] - cornerAlignB[0][0]) * (cornerAlignB[1][1] - cornerAlignB[0][1]);
    
    float interW = std::min(cornerAlignA[1][0], cornerAlignB[1][0]) - std::max(cornerAlignA[0][0], cornerAlignB[0][0]);
    float interH = std::min(cornerAlignA[1][1], cornerAlignB[1][1]) - std::max(cornerAlignA[0][1], cornerAlignB[0][1]);
    
    float sInter = std::max(interW, 0.0f) * std::max(interH, 0.0f);
    float sUnion = sBoxA + sBoxB - sInter;
    
    return sInter/sUnion;
}

static void AlignedNMSBev(std::vector<Box>& predBoxs){
    
    if(predBoxs.size() == 0)
        return;

    std::sort(predBoxs.begin(),predBoxs.end(),[ ](Box& box1, Box& box2){return box1.score > box2.score;});

    auto boxSize = predBoxs.size() > INPUT_NMS_MAX_SIZE? INPUT_NMS_MAX_SIZE : predBoxs.size();
    
    for(auto boxIdx1 =0U; boxIdx1 < boxSize; boxIdx1++){
        for(auto boxIdx2 = boxIdx1+1; boxIdx2 < boxSize; boxIdx2++){
            if(predBoxs[boxIdx2].isDrop == true)
                continue;
            if(IoUBev(predBoxs[boxIdx1], predBoxs[boxIdx2]) > NMS_THREAHOLD)
                predBoxs[boxIdx2].isDrop = true;
        }
    }
}

void NuscenesDetectionOnnxApplication::postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch)
{
    std::vector<std::string> regName{   "594", "618", "642", "666", "690", "714"};
    std::vector<std::string> heightName{"598", "622", "646", "670", "694", "718"};
    std::vector<std::string> rotName{   "606", "630", "654", "678", "702", "726"};
    std::vector<std::string> velName{   "610", "634", "658", "682", "706", "730"};
    std::vector<std::string> dimName{   "736", "740", "744", "748", "752", "756"};
    std::vector<std::string> scoreName{ "737", "741", "745", "749", "753", "757"};
    std::vector<std::string> clsName{   "738", "742", "746", "750", "754", "758"};
    int clsOffsetPerTask[] = {0, 1, 3, 5, 6, 8};
	std::vector<Box> predResult;
	int lidar_index = (sample_index * batch) % dataset->getSize();
	
	for (size_t taskIdx = 0; taskIdx < TASK_NUM; taskIdx++){
        std::vector<Box> predBoxs;

		float* reg = static_cast<float*>(output_buffers[outputIndexMap[regName[taskIdx]]]);
        float* height = static_cast<float*>(output_buffers[outputIndexMap[heightName[taskIdx]]]);
        float* rot = static_cast<float*>(output_buffers[outputIndexMap[rotName[taskIdx]]]);
        float* vel = static_cast<float*>(output_buffers[outputIndexMap[velName[taskIdx]]]);
        float* dim = static_cast<float*>(output_buffers[outputIndexMap[dimName[taskIdx]]]);
        float* score = static_cast<float*>(output_buffers[outputIndexMap[scoreName[taskIdx]]]);
        int32_t* cls = (int32_t *) output_buffers[outputIndexMap[clsName[taskIdx]]];

		for(size_t yIdx=0; yIdx < OUTPUT_H; yIdx++){
            for(size_t xIdx=0; xIdx < OUTPUT_W; xIdx++){
                auto idx = yIdx* OUTPUT_W + xIdx;
                if(score[idx] < SCORE_THRESHOLD)
                    continue;
                
                float x = (xIdx + reg[0*OUTPUT_H*OUTPUT_W + idx])*OUT_SIZE_FACTOR*X_STEP + X_MIN;
                float y = (yIdx + reg[1*OUTPUT_H*OUTPUT_W + idx])*OUT_SIZE_FACTOR*Y_STEP + Y_MIN;
                float z = height[idx];

                if(x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX || z < Z_MIN || z > Z_MAX)
                    continue;
                
                Box box;
                box.x = x;
                box.y = y;
                box.z = z;
                box.l = dim[0*OUTPUT_H*OUTPUT_W + idx];
                box.h = dim[1*OUTPUT_H*OUTPUT_W + idx];
                box.w = dim[2*OUTPUT_H*OUTPUT_W + idx];
                box.theta = atan2(rot[0*OUTPUT_H*OUTPUT_W + idx], rot[1*OUTPUT_H*OUTPUT_W + idx]);
                box.velX = vel[0*OUTPUT_H*OUTPUT_W+idx];
                box.velY = vel[1*OUTPUT_H*OUTPUT_W+idx];
                // box.theta = box.theta - PI /2;

                box.score = score[idx];
                box.cls = cls[idx] + clsOffsetPerTask[taskIdx];
                box.isDrop = false;

                predBoxs.push_back(box);

            }
        }
        
        AlignedNMSBev(predBoxs);

        for(auto idx =0U; idx < predBoxs.size(); idx++){
            if(!predBoxs[idx].isDrop)
                predResult.push_back(predBoxs[idx]);
        }
	}

    result_format->saveOutput(predResult, dataset->getData(lidar_index)->path);
}

void NuscenesDetectionOnnxApplication::postprocessing2(int thread_id, int sample_index, int batch) {
	// do nothing
}

void NuscenesDetectionOnnxApplication::writeResultFile(std::string result_file_name) {
	result_format->writeResultFile(result_file_name);
}

NuscenesDetectionOnnxApplication::~NuscenesDetectionOnnxApplication()
{
	labels.clear();
	delete dataset;
	//delete result_format;
}
