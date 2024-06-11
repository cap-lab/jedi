
#include <string.h>

#include "lidar_dataset.h"
#include "pillar.h"
#include "int8_pillar_calibrator.h"

Int8PillarEntropyCalibrator::Int8PillarEntropyCalibrator(nvinfer1::INetworkDefinition *network, const std::string& fileLidarlist, const int calib_lidar_num, 
                            const std::string calibTableFilePath, bool readCache):  mCalibTableFilePath(calibTableFilePath), mReadCache(readCache), mCalibLidarNum(calib_lidar_num) {
    
    int num_inputs = network->getNbInputs();
	for (int i = 0 ; i < num_inputs ; i++) {
		nvinfer1::ITensor *tensor_input = network->getInput(i);
        nvinfer1::DataType type = tensor_input->getType();
        nvinfer1::Dims tensor_dim = tensor_input->getDimensions();
        void *bufPtr = nullptr;
        int input_size = 1;

        for(int dim = 0 ; dim < tensor_dim.nbDims ; dim++) {
            input_size *= tensor_dim.d[dim];
        }

        if(type == nvinfer1::DataType::kFLOAT) {
            input_size *= sizeof(float);
        }
        else if(type == nvinfer1::DataType::kINT32) {
            input_size *= sizeof(int);
        }
        else if(type == nvinfer1::DataType::kHALF) {
            input_size *= sizeof(short);
        }

        checkCuda(cudaMalloc(&bufPtr, input_size));

        std::cout << "calib tensor print: " << tensor_input->getName() << ", buf_ptr: " << bufPtr << ", size: " << input_size << std::endl;

        bindingMap[tensor_input->getName()] = bufPtr;

        if(strcmp(tensor_input->getName(),"onnx::MatMul_0") == 0) {
            featureBuf = (float *) malloc(input_size);
            if (featureBuf == nullptr) {
                 exit(EXIT_FAILURE);
            }
            featureSize = input_size;
        } else if(strcmp(tensor_input->getName(),"indices_input") == 0) {
            indiceBuf = (int *) malloc(input_size);
            if (indiceBuf == nullptr) {
                 exit(EXIT_FAILURE);
            }
            indiceSize = input_size;
        }
	}
    calibLidarSet = new LidarDataset(fileLidarlist);
    calibLidarIndex = 0;
}

bool Int8PillarEntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) NOEXCEPT {
    int point_num = 0;
    bool readBinOk = false;

    std::cout << "calib index: "<< calibLidarIndex << std::endl;

    if (calibLidarIndex >= mCalibLidarNum)
        return false;

    readBinOk = readBinFile(calibLidarSet->getData(calibLidarIndex)->path, fileInputBuf, point_num, fileInputBufSize);
    if(readBinOk == false) {
        exit(EXIT_FAILURE);
    }

    memset(indiceBuf, -1, indiceSize);
    memset(featureBuf, 0, featureSize);

    makePillars(fileInputBuf, featureBuf, indiceBuf, point_num, 0, MAX_PILLARS);

    for(int i = 0 ; i < nbBindings ; i++) {
        if(strcmp(names[i], "onnx::MatMul_0") == 0) {
            checkCuda(cudaMemcpy(bindingMap[names[i]], featureBuf, featureSize, cudaMemcpyHostToDevice));
            std::cout << "calib feature print("<< i <<"): " << names[i] << ", binding_ptr: " << bindingMap[names[i]] << ", size: " << featureSize << std::endl;
            bindings[i] = bindingMap[names[i]];
        }
        else if(strcmp(names[i], "indices_input") == 0) {
            checkCuda(cudaMemcpy(bindingMap[names[i]], indiceBuf, indiceSize, cudaMemcpyHostToDevice));
            std::cout << "calib indices print("<< i <<"): " << names[i] << ", binding_ptr: " << bindingMap[names[i]] << ", size: " << indiceSize << std::endl;
            bindings[i] = bindingMap[names[i]];
        }
    }
    
    calibLidarIndex++;

    return true;
}

const void* Int8PillarEntropyCalibrator::readCalibrationCache(size_t& length) NOEXCEPT {
    mCalibrationCache.clear();
    assert(!mCalibTableFilePath.empty());
    std::ifstream input(mCalibTableFilePath, std::ios::binary);
    input >> std::noskipws;
    input >> std::noskipws;
    if (mReadCache && input.good())
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                  std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8PillarEntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) NOEXCEPT {
    assert(!mCalibTableFilePath.empty());
    std::ofstream output(mCalibTableFilePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}
