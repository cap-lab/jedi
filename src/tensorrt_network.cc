
#include <iostream>
#include <string>
#include <cstring>

#include "tensorrt_network.h"


using namespace nvinfer1;

static void printCenteredTitle(const char *title, char fill, int dim) {

	int len = strlen(title);
	int first = dim/2 + len/2;
		
	if(len >0) 
		std::cout<<"\n";
	std::cout.width(first); std::cout.fill(fill); std::cout<<std::right<<title;
	std::cout.width(dim - first); std::cout<<"\n"; 
	std::cout.fill(' ');
}


static std::string convertLayerTypeToString(ILayer *layer) {
	std::string type_name;
	switch(layer->getType()){
		case LayerType::kCONVOLUTION:
			return "Convolution";
		case LayerType::kFULLY_CONNECTED:
			return "Fully Connected";
		case LayerType::kACTIVATION: {
			IActivationLayer *actlayer = (IActivationLayer *) layer;
				switch(actlayer->getActivationType()) {
					case ActivationType::kRELU:
						return "ReLU";
					case ActivationType::kSIGMOID:
						return "Sigmoid";
					case ActivationType::kTANH:
						return "TanH";
					case ActivationType::kLEAKY_RELU:
						return "LeakyReLU";
					case ActivationType::kELU:
						return "ELU";
					case ActivationType::kSELU:
						return "SELU";
					case ActivationType::kSOFTSIGN:
						return "SoftSign";
					case ActivationType::kSOFTPLUS:
						return "SoftPlus";
					case ActivationType::kCLIP:
						return "Clip";
					case ActivationType::kHARD_SIGMOID:
						return "HardSigmoid";
					case ActivationType::kSCALED_TANH:
						return "ScaledTanH";
					case ActivationType::kTHRESHOLDED_RELU:
						return "ThresholdedReLU";
				}	
			}
			return "UnknownActivation";
		case LayerType::kPOOLING: {
				IPoolingLayer *poolLayer = (IPoolingLayer *) layer;
				switch(poolLayer->getPoolingType()) {
					case PoolingType::kMAX:
						return "MaxPooling";
					case PoolingType::kAVERAGE:
						return "AveragePooling";
					case PoolingType::kMAX_AVERAGE_BLEND:	
						return "MaxAvgPooling";
				}
			}
			return "UnknownPooling";
		case LayerType::kLRN:
			return "LRN";
		case LayerType::kSCALE:
			return "Scale";
		case LayerType::kSOFTMAX:
			return "SoftMax";
		case LayerType::kDECONVOLUTION:
			return "Deconvolution";
		case LayerType::kCONCATENATION:
			return "Concatenation";
		case LayerType::kELEMENTWISE:
			return "Elementwise";
		case LayerType::kPLUGIN:
			return "Plugin";
		case LayerType::kUNARY:
			return "UnaryOp";
		case LayerType::kPADDING:
			return "Padding";
		case LayerType::kSHUFFLE:
			return "Shuffle";
		case LayerType::kREDUCE:
			return "Reduce";
		case LayerType::kTOPK:
			return "TopK";
		case LayerType::kGATHER:
			return "Gather";
		case LayerType::kMATRIX_MULTIPLY:
			return "Matrix Multiply";
		case LayerType::kRAGGED_SOFTMAX:
			return "Ragged Softmax";
		case LayerType::kCONSTANT:
			return "Constant";
		case LayerType::kRNN_V2:
			return "RNNv2";
		case LayerType::kIDENTITY:
			return "Identity";
		case LayerType::kPLUGIN_V2:
			return "PluginV2";
		case LayerType::kSLICE:
			return "Slice";
		case LayerType::kSHAPE:
			return "Shape";
		case LayerType::kPARAMETRIC_RELU:
			return "Parametric ReLU";
		case LayerType::kRESIZE:
			return "Resize";
		case LayerType::kTRIP_LIMIT:
			return "Loop Trip limit";
		case LayerType::kRECURRENCE:
			return "Loop Recurrence";
		case LayerType::kITERATOR:
			return "Loop Iterator";
		case LayerType::kLOOP_OUTPUT:
			return "Loop output";
		case LayerType::kSELECT:
			return "Select";
		case LayerType::kFILL:
			return "Fill";
		case LayerType::kQUANTIZE:
			return "Quantize";
		case LayerType::kDEQUANTIZE:
			return "Dequantize";
		case LayerType::kCONDITION:
			return "Condition";
		case LayerType::kCONDITIONAL_INPUT:
			return "Conditional Input";
		case LayerType::kCONDITIONAL_OUTPUT:
			return "Conditional Output";
		case LayerType::kSCATTER:
			return "Scatter";
		case LayerType::kEINSUM:
			return "Einsum";
		case LayerType::kASSERTION:
			return "Assertion";
		case LayerType::kONE_HOT:
			return "OneHot";
		case LayerType::kNON_ZERO:
			return "NonZero";
		case LayerType::kGRID_SAMPLE:
			return "Grid Sample";
		case LayerType::kNMS:
			return "NMS";
	}

	return "Unknown";
}

void TensorRTNetwork::printNetwork() {
	if(network != nullptr) {
		int layer_num = network->getNbLayers();
		std::cout.width(4); std::cout << std::left << "N.";
		std::cout<<" ";
		std::cout.width(17); std::cout<<std::left<<"Layer type";
		std::cout<<" ";
		std::cout.width(17); std::cout<<std::left<<"Layer name";
		std::cout<<" ";
		//std::cout.width(22); std::cout<<std::left<<"input (H*W,CH)";
		//std::cout.width(16); std::cout<<std::left<<"output (H*W,CH)";
		std::cout << std::endl;

		printCenteredTitle("", '=', 60);

		for (int i = 0 ; i < layer_num ; i++) {
			ILayer *layer = network->getLayer(i);

			std::cout.width(4); std::cout<<std::right<<i;
			std::cout<<" ";

			std::cout.width(16); std::cout<<std::left<< convertLayerTypeToString(layer);
			std::cout<<" ";

			std::cout.width(16); std::cout<<std::left<< layer->getName();
			std::cout<<" ";
			std::cout << std::endl;

//			std::cout<<"\t input num: "<<layer->getNbInputs()<<", output num: "<<layer->getNbOutputs()<<std::endl;
//
//			for (int j = 0; j < layer->getNbInputs(); j++) {
//				ITensor *tensor = layer->getInput(j);	
//				if(tensor != nullptr) {
//					std::cout.width(20); std::cout<<std::left<<"\t"<<tensor->getName();
//					std::cout<<" ";
//					std::cout << std::endl;
//				}
//			}
//
//			std::cout.width(20); std::cout<<std::left<<"\t-------";
//			std::cout<<" ";
//			std::cout << std::endl;
//
//			for (int j = 0; j < layer->getNbOutputs(); j++) {
//				ITensor *tensor = layer->getOutput(j);	
//				if(tensor != nullptr) {
//					std::cout.width(20); std::cout<<std::left<<"\t"<<tensor->getName();
//					std::cout<<" ";
//					std::cout << std::endl;
//				}
//			}
		}
	}
}


