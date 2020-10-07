#include <iostream>
#include <vector>
#include <cassert>

#include <NvInfer.h>
#include <tkDNN/tkdnn.h>
#include <tkDNN/DarknetParser.h>

#include "cuda.h"

#include "model.h"
#include "variable.h"

Model::Model(ConfigData *config_data, int instance_id) {
	this->config_data = config_data;
	this->instance_id = instance_id;
}

Model::~Model() {
	int device_num = config_data->instances.at(instance_id).device_num;
	
	start_bindings.clear();
	binding_size.clear();
	netRTs.clear();
	is_net_output.clear();
	stream_buffers.clear();
	input_buffers.clear();
	output_buffers.clear();

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		contexts.at(iter1).clear();	
		streams.at(iter1).clear();
	}
}

void Model::getModelFileName(int curr, char *fileName) {
	char cut_points_name[STRING_SIZE];
	char device_name[STRING_SIZE];
	int device = config_data->instances.at(instance_id).devices.at(curr);
	int batch = config_data->instances.at(instance_id).batch;
	int prev_cut_point = 0, curr_cut_point = 0;
	
	if(curr > 0) {
		prev_cut_point = config_data->instances.at(instance_id).cut_points.at(curr-1);
	}
	curr_cut_point = config_data->instances.at(instance_id).cut_points.at(curr);

	snprintf(cut_points_name, STRING_SIZE, "%d.%d", prev_cut_point, curr_cut_point);

	if(device == DEVICE_DLA) {
		snprintf(device_name, STRING_SIZE, "DLA");
	}
	else {
		snprintf(device_name, STRING_SIZE, "GPU");
	}

	snprintf(fileName, STRING_SIZE, "%smodel_%s_%s_FP16_%d.model", config_data->instances.at(instance_id).model_dir.c_str(), cut_points_name, device_name, batch);
}

void Model::setDevice(int curr) {
	int device = config_data->instances.at(instance_id).devices.at(curr);

	if(device == DEVICE_DLA) {
		net->dla = true;	
	}
	else {
		net->dla = false;	
	}
}

void Model::setMaxBatchSize() {
	int batch = config_data->instances.at(instance_id).batch;

	net->maxBatchSize = batch;
}

void Model::initializeModel() {
	std::string bin_path(config_data->instances.at(instance_id).bin_path);
    std::string wgs_path  = bin_path + "/layers";
    std::string cfg_path(config_data->instances.at(instance_id).cfg_path);
    std::string name_path(config_data->instances.at(instance_id).name_path); 
	int device_num = config_data->instances.at(instance_id).device_num;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int start_index = 0;

	// parse a network using tkDNN darknetParser
	net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);
	net->print();

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		char fileName[2 * STRING_SIZE];
		int cut_point = config_data->instances.at(instance_id).cut_points[iter1];
		int dla_core = config_data->instances.at(instance_id).dla_cores[iter1];

		getModelFileName(iter1, fileName);

		setDevice(iter1);
		net->fp16 = true;
		net->int8 = false;
		setMaxBatchSize();

		tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, (const char*)fileName, start_index, cut_point, dla_core);
		assert(netRT->engineRT != nullptr);

		netRTs.push_back(netRT);
	
		start_index = cut_point + 1;
	}
	net->releaseLayers();
	delete net;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		std::vector<nvinfer1::IExecutionContext *> context_vec;
		for(int iter2 = 0; iter2 < buffer_num; iter2++) {
			nvinfer1::IExecutionContext *context = netRTs[iter1]->engineRT->createExecutionContext();	
			assert(context);

			context_vec.push_back(context);
		}
		contexts.push_back(context_vec);
	}
}

void Model::finalizeModel() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		for(int iter2 = 0; iter2 < buffer_num; iter2++) {
			nvinfer1::IExecutionContext *context = contexts[iter1][iter2];		
			context->destroy();
		}
		delete netRTs[iter1];
	}
}

void Model::setBindingsNum(int curr, int &input_binding_num, int &output_binding_num) {
	input_binding_num = 0;
	output_binding_num = 0;

	for(int iter = 0; iter < netRTs[curr]->engineRT->getNbBindings(); iter++) {
		if(netRTs[curr]->engineRT->bindingIsInput(iter)) {
			input_binding_num++;
		}	
		else {
			output_binding_num++;	
		}
	}
}

void Model::initializeBindingVariables() {
	int device_num = config_data->instances.at(instance_id).device_num;

	start_bindings.push_back(1);
	for(int iter = 1; iter <= device_num; iter++) {
		start_bindings.push_back(-1);
	}

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		int curr_binding_num = netRTs[iter1]->engineRT->getNbBindings();
		for(int iter2 = 0; iter2 < curr_binding_num; iter2++) {
			is_net_output.push_back(false);
			binding_size.push_back(0);
		}	
	}
	binding_size.at(0) = INPUT_SIZE;

	yolo_num = 0;
	total_binding_num = 1;
	output_num = 0;
}

void Model::setBufferIndexing() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int input_binding_num = 0, output_binding_num = 0, curr_binding_num = 0;
	
	initializeBindingVariables();

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		setBindingsNum(iter1, input_binding_num, output_binding_num);
		curr_binding_num = netRTs[iter1]->engineRT->getNbBindings();

		start_bindings[iter1] = start_bindings[iter1] - input_binding_num;
		start_bindings[iter1+1] = start_bindings[iter1] + curr_binding_num;


		for(int iter2 = input_binding_num; iter2 < curr_binding_num; iter2++) {
			nvinfer1::Dims dim = netRTs[iter1]->engineRT->getBindingDimensions(iter2);	
			int index = start_bindings[iter1];
			binding_size[index + iter2] = dim.d[0] * dim.d[1] * dim.d[2];
			fprintf(stderr, "dim.d[0]: %d, dim.d[1]: %d, dim.d[2]: %d\n", dim.d[0], dim.d[1], dim.d[2]);
			total_binding_num++;
		}

		for(int iter2 = 0; iter2 < netRTs[iter1]->pluginFactory->n_yolos; iter2++) {
			YoloData yolo;
			yolo.n_masks = netRTs[iter1]->pluginFactory->yolos[yolo_num + iter2]->n_masks;	
			yolo.bias = netRTs[iter1]->pluginFactory->yolos[yolo_num + iter2]->bias;	
			yolo.mask = netRTs[iter1]->pluginFactory->yolos[yolo_num + iter2]->mask;	

			yolos.push_back(yolo);
		}	
		yolo_num += netRTs[iter1]->pluginFactory->n_yolos;

		int index = start_bindings[iter1] + curr_binding_num - output_binding_num;
		for(int iter2 = index; iter2 < index + netRTs[iter1]->pluginFactory->n_yolos; iter2++) {
			is_net_output[iter2] = true;
			output_num++;
		}
	}
}

void Model::allocateStream() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		std::vector<cudaStream_t> stream_vec;
		for(int iter2 = 0; iter2 < buffer_num; iter2++) {
			cudaStream_t stream;
			check_error(cudaStreamCreate(&stream));

			stream_vec.push_back(stream);
		}
		streams.push_back(stream_vec);
	}
}

void Model::deallocateStream() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		for(int iter2 = 0; iter2 < buffer_num; iter2++) {
			cudaStream_t stream = streams[iter1].back();
			cudaStreamDestroy(stream);

			streams[iter1].pop_back();
		}
	}
}

void Model::setStreamBuffer() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int iter1 = 0; iter1 < total_binding_num; iter1++) {
		for(int iter2 = 0; iter2 < buffer_num; iter2++) {
			void *tmp;
			stream_buffers.push_back(tmp);	
		}
	}
}

void Model::allocateBuffer() {
	int batch = config_data->instances.at(instance_id).batch;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int iter1 = 0; iter1 < buffer_num; iter1++) {
		float *input_buffer = cuda_make_array_host(batch * binding_size[0]);
		cudaHostGetDevicePointer(&(stream_buffers[iter1 * total_binding_num]), input_buffer, 0);
		input_buffers.push_back(input_buffer);

		if(total_binding_num > 2) {
			for(int iter2 = 1; iter2 < total_binding_num-1; iter2++) {
				if(!is_net_output[iter2]) {
					stream_buffers[iter1 * total_binding_num + iter2] = cuda_make_array_16(NULL, batch * binding_size[iter2]);	
				}
				else {
					float *middle_buffer = cuda_make_array_host(batch * binding_size[iter2]);
					cudaHostGetDevicePointer(&(stream_buffers[iter1 * total_binding_num + iter2]), middle_buffer, 0);
					output_buffers.push_back(middle_buffer);
				}
			}	
		}

		float *output_buffer = cuda_make_array_host(batch * binding_size[total_binding_num - 1]);
		cudaHostGetDevicePointer(&(stream_buffers[total_binding_num - 1 + iter1 * total_binding_num]), output_buffer, 0);
		output_buffers.push_back(output_buffer);
	}
}

void Model::deallocateBuffer() {
	while(!input_buffers.empty()) {
		float *buffer = input_buffers.back();	
		cudaFreeHost(buffer);
		input_buffers.pop_back();
	}

	while(!output_buffers.empty()) {
		float *buffer = output_buffers.back();	
		cudaFreeHost(buffer);
		output_buffers.pop_back();
	}
}

void Model::initializeBuffers() {
	setBufferIndexing();	
	allocateStream();
	setStreamBuffer();
	allocateBuffer();
}

void Model::finalizeBuffers() {
	deallocateBuffer();
	deallocateStream();
}

bool Model::checkInferenceDone(int device_id, int buffer_id) {
	return cudaStreamQuery(streams[device_id][buffer_id]) == cudaSuccess;	
}

void Model::infer(int device_id, int buffer_id) {
	int start_binding = start_bindings[device_id] + total_binding_num * buffer_id;
	int batch = config_data->instances.at(instance_id).batch;

	contexts[device_id][buffer_id]->enqueue(batch, &(stream_buffers[start_binding]), streams[device_id][buffer_id], nullptr);
	// contexts[device_id][buffer_id]->execute(batch, &(stream_buffers[start_binding]));
}

void Model::waitUntilInferenceDone(int device_id, int buffer_id) {
	cudaStreamSynchronize(streams[device_id][buffer_id]);
}
