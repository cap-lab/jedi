#include <iostream>
#include <vector>
#include <cassert>
#include <unistd.h>

#include "variable.h"
#include "config.h"
#include "dataset.h"
#include "thread.h"
#include "model.h"
#include "image_opencv.h"
#include "region_wrapper.h"
#include "yolo_wrapper.h"
#include "coco.h"

#define MAX_TIMEOUT (100000)

static int readImage(float *input_buffer, Dataset *dataset, InputDim input_dim, int batch, int pre_thread_num, int index) {
	for(int iter = 0; iter < batch; iter++) {
		int orignal_width = 0, original_height = 0;
		int input_width = input_dim.width, input_height = input_dim.height, input_channel = input_dim.channel;
		int input_size = input_width * input_height * input_channel;

		loadImageResize((char *)(dataset->paths[index + iter].c_str()), input_width, input_height, input_channel, &orignal_width, &original_height, input_buffer + iter * input_size);	
		dataset->w.at(index + iter) = orignal_width;
		dataset->h.at(index + iter) = original_height;
	}	

	return index + batch * pre_thread_num;
}

void doPreProcessing(void *d) {
	PreProcessingThreadData *data = (PreProcessingThreadData *)d;
	ConfigData *config_data = data->config_data;
	int instance_id = data->instance_id;
	int tid = data->tid;
	int *signals = data->signals;
	Dataset *dataset = data->dataset;

	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int batch = config_data->instances.at(instance_id).batch;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int pre_thread_num = config_data->instances.at(instance_id).pre_thread_num;
	int sample_index = sample_offset + tid;
	int index = (sample_offset + tid) * batch;
	long stuckWhile = 0;

	while(sample_index < sample_offset + sample_size && exit_flag == false) {
		while(signals[sample_index % buffer_num] && exit_flag == false) {
			usleep(SLEEP_TIME);	
			stuckWhile++;
		}

		index = readImage(data->model->input_buffers.at(sample_index % buffer_num), dataset, data->model->input_dim, batch, pre_thread_num, index);

		signals[sample_index % buffer_num] = 1;
		sample_index += pre_thread_num;
	}

	fprintf(stderr, "stuckWhile(front thread: %d): %ld\n", tid, stuckWhile);
}

static void detectBox(std::vector<float *> output_buffers, int buffer_id, std::vector<YoloData> yolos, InputDim input_dim, int batch, std::string network_name, Detection *dets, std::vector<int> &detections_num) { if(network_name == NETWORK_YOLOV2 || network_name == NETWORK_YOLOV2TINY || network_name == NETWORK_DENSENET) {
		regionLayerDetect(input_dim, batch, output_buffers.at(buffer_id), dets, &(detections_num[0]));	
	}
	else {
		yoloLayerDetect(input_dim, batch, output_buffers, buffer_id, yolos, dets, detections_num);
	}
}

static void printBox(Dataset *dataset, int sample_index, InputDim input_dim, int batch, std::string network_name, Detection *dets, std::vector<int> detections_num) {
	for(int iter1 = 0; iter1 < batch; iter1++) {
		int index = sample_index * batch + iter1;

		if(network_name == NETWORK_YOLOV2 || network_name == NETWORK_YOLOV2TINY || network_name == NETWORK_DENSENET) {
			printDetector(input_dim, &dets[iter1 * NBOXES], index, dataset, detections_num[0]);
		}
		else {
			printDetector(input_dim, &dets[iter1 * NBOXES], index, dataset, detections_num[iter1]);
		}
	}
}

void doPostProcessing(void *d) {
	PostProcessingThreadData *data = (PostProcessingThreadData *)d;
	ConfigData *config_data = data->config_data;
	int instance_id = data->instance_id;
	int tid = data->tid;
	int *signals = data->signals;
	Dataset *dataset = data->dataset;

	int instance_num = config_data->instance_num;
	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int batch = config_data->instances.at(instance_id).batch;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int post_thread_num = config_data->instances.at(instance_id).post_thread_num;
	std::string network_name = config_data->instances.at(instance_id).network_name;
	int sample_index = sample_offset + tid;
	std::vector<YoloData> yolos = data->model->yolos;
	int buffer_id = 0;
	long stuckWhile = 0;

	Detection *dets;
	std::vector<int> detections_num(batch, 0);

	buffer_id = sample_index % buffer_num;

	setBiases(network_name);
	allocateDetectionBox(batch, &dets);

	while(sample_index < sample_offset + sample_size && exit_flag == false) {
		while(!signals[sample_index % buffer_num] && exit_flag == false) {
			usleep(SLEEP_TIME);	
			stuckWhile++;
		}	

		buffer_id = sample_index % buffer_num; 

		detectBox(data->model->output_buffers, buffer_id, yolos, data->model->input_dim, batch, network_name, dets, detections_num);

		signals[sample_index % buffer_num] = 0;

		printBox(dataset, sample_index, data->model->input_dim, batch, network_name, dets, detections_num);
		
		if(tid == (sample_index % post_thread_num) && instance_id == 0) {
			std::cerr<<"[TEST | "<<(sample_index+1)*instance_num<<" / "<<sample_size*instance_num<<"]\r";	
		}

		sample_index += post_thread_num;
	}

	fprintf(stderr, "stuckWhile(back thread: %d): %ld\n", tid, stuckWhile);

	deallocateDetectionBox(batch * NBOXES, dets);
}

void doInference(void *d) {
	InferenceThreadData *data = (InferenceThreadData *)d;
	ConfigData *config_data = data->config_data;
	int instance_id = data->instance_id;
	int device_id = data->tid;
	int *curr_signals = data->curr_signals;
	int *next_signals = data->next_signals;
	Model *model = data->model;

	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	std::string network_name = config_data->instances.at(instance_id).network_name;
	int sample_index = sample_offset;
	std::vector<int> ready(buffer_num, 1);
	int sleep_time = 0;
	long stuckWhile = 0;

	while(sample_index < sample_offset + sample_size && exit_flag == false) {
		while(exit_flag == false) {
			if(curr_signals[sample_index % buffer_num] == 1 && next_signals[sample_index % buffer_num] == 0 && ready[sample_index % buffer_num] == 1) {
				break;	
			}

			for(int iter = 0; iter < buffer_num; iter++) {
				if(ready[iter] == 0) {
					if(model->checkInferenceDone(device_id, iter)) {
						curr_signals[iter] = 0;	
						next_signals[iter] = 1;
						ready[iter] = 1;
					}
				}	
			}
			usleep(SLEEP_TIME);
			sleep_time++;
			stuckWhile++;

			if(sleep_time > MAX_TIMEOUT) {
				exit_flag = true;
				printf("timeout is reached. program will be terminated.\n");
			}
		}	
		
		sleep_time = 0;
		model->infer(device_id, sample_index % buffer_num);
		ready[sample_index % buffer_num] = 0;

		sample_index++;
	}
	fprintf(stderr, "stuckWhile(device id: %d): %ld\n", device_id, stuckWhile);

	for(int iter = 0; iter < buffer_num; iter++) {
		if(ready[iter] == 0) {
			if(exit_flag == false) 
			{
				model->waitUntilInferenceDone(device_id, iter);
			}
			curr_signals[iter] = 0;
			next_signals[iter] = 1;
			ready[iter] = 1;
		}	
	}
}
