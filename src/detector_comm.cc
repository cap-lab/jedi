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

std::vector<long> pre_time_vec, post_time_vec;
extern int g_pre_core, g_post_core;
extern bool exit_flag;

static long getTime() {
	struct timespec time;
	if(0 != clock_gettime(CLOCK_REALTIME, &time)) {
		std::cerr<<"Something wrong on clock_gettime()"<<std::endl;		
		exit(-1);
	}
	return (time.tv_nsec) / 1000 + time.tv_sec * 1000000; // us
}

static int stickThisThreadToCore(int core_id) {
	int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
	if (core_id < 0 || core_id >= num_cores)
		return EINVAL;

	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core_id, &cpuset);

	pthread_t current_thread = pthread_self();    
	return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

static int readImage(float *input_buffer, Dataset *dataset, InputDim input_dim, int batch, int pre_thread_num, int index) {
	for(int iter = 0; iter < batch; iter++) {
		// int orignal_width = 0, original_height = 0;
		int input_width = input_dim.width, input_height = input_dim.height, input_channel = input_dim.channel;
		int input_size = input_width * input_height * input_channel;
		int image_index = (index + iter) % dataset->m;

		// loadImageResize((char *)(dataset->paths[image_index].c_str()), input_width, input_height, input_channel, &orignal_width, &original_height, input_buffer + iter * input_size);	
		// dataset->w.at(image_index) = orignal_width;
		// dataset->h.at(image_index) = original_height;
		
		loadImage((char *)(dataset->paths[image_index].c_str()), input_size, input_buffer + iter * input_size);	
	}	

	return index + batch * pre_thread_num;
}

void doPreProcessing(void *d) {
	PreProcessingThreadData *data = (PreProcessingThreadData *)d;
	ConfigData *config_data = data->config_data;
	int instance_id = data->instance_id;
	int tid = data->tid;
	std::vector<int> *signals = data->signals;
	Dataset *dataset = data->dataset;

	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int batch = config_data->instances.at(instance_id).batch;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int pre_thread_num = config_data->instances.at(instance_id).pre_thread_num;
	int sample_index = sample_offset + tid;
	int index = (sample_offset + tid) * batch;
	long curr_time = getTime();
	int prev_pre_core = -1;


	while(sample_index < sample_offset + sample_size) {
		while((*signals)[sample_index % buffer_num]) {
			usleep(SLEEP_TIME);	
		}

		if(prev_pre_core != g_pre_core) {
			stickThisThreadToCore(g_pre_core);
			prev_pre_core = g_pre_core;
		}

		curr_time = getTime();

		if(exit_flag) {
			break;	
		}

		index = readImage(data->model->input_buffers.at(sample_index % buffer_num), dataset, data->model->input_dim, batch, pre_thread_num, index);

		(*signals)[sample_index % buffer_num] = 1;
		sample_index += pre_thread_num;
		pre_time_vec.push_back(getTime() - curr_time);
	}
}

static void detectBox(std::vector<float *> output_buffers, int buffer_id, std::vector<YoloData> yolos, InputDim input_dim, int batch, std::string network_name, Detection *dets, std::vector<int> &detections_num) {
	if(network_name == NETWORK_YOLOV2 || network_name == NETWORK_YOLOV2TINY || network_name == NETWORK_DENSENET) {
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
	std::vector<int> *signals = data->signals;
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
	Detection *dets;
	std::vector<int> detections_num(batch, 0);
	long curr_time = getTime();
	int prev_post_core = -1;

	buffer_id = sample_index % buffer_num;

	setBiases(network_name);
	allocateDetectionBox(batch, &dets);

	while(sample_index < sample_offset + sample_size) {
		while(!(*signals)[sample_index % buffer_num]) {
			usleep(SLEEP_TIME);	
		}	

		if(prev_post_core != g_post_core) {
			stickThisThreadToCore(g_post_core);
			prev_post_core = g_post_core;
		}

		curr_time = getTime();

		if(exit_flag) {
			break;	
		}

		buffer_id = sample_index % buffer_num; 

		detectBox(data->model->output_buffers, buffer_id, yolos, data->model->input_dim, batch, network_name, dets, detections_num);

		(*signals)[sample_index % buffer_num] = 0;

		printBox(dataset, sample_index, data->model->input_dim, batch, network_name, dets, detections_num);
		
		if(tid == (sample_index % post_thread_num) && instance_id == 0) {
			std::cerr<<"[TEST | "<<(sample_index+1)*instance_num<<" / "<<sample_size*instance_num<<"]\r";	
		}

		sample_index += post_thread_num;
		post_time_vec.push_back(getTime() - curr_time);
	}

	deallocateDetectionBox(batch * NBOXES, dets);
}
