#include <iostream>
#include <vector>
#include <cassert>
#include <unistd.h>
#include <list>
#include <mutex>

#include "variable.h"
#include "config.h"
#include "dataset.h"
#include "thread.h"
#include "model.h"
#include "image_opencv.h"
#include "region_wrapper.h"
#include "yolo_wrapper.h"
#include "coco.h"
#include "util.h"

#include "inference_application.h"

#define MAX_TIMEOUT (100000)

long getAverageLatency(int instance_id, ConfigData *config_data, std::vector<long> latency)
{
	long sum  = 0;
	int nSize = latency.size(); 

	for(int iter = 0 ; iter < nSize ; iter++) {
		sum += latency[iter];
	}

	return sum / (long) nSize;
}

static void readData(int thread_id, float *input_buffer, IInferenceApplication *app, int total_input_size, int batch, int batch_thread_num, int index)
{
	int input_size = total_input_size / batch;
	#pragma omp parallel num_threads(batch_thread_num)
	#pragma omp for
	for(int iter = 0; iter < batch; iter++) {
		app->preprocessing(thread_id, index, iter, input_buffer + iter * input_size);
	}
}

static int getMinSampleIndex(std::vector<int> *running_list, int cur_sample_index)
{
	int size = running_list->size();
	int min_sample_index = cur_sample_index;
	for (int iter = 0; iter < size ; iter++) {
		if(min_sample_index > (*running_list)[iter]) {
			min_sample_index = (*running_list)[iter];
		}
	}
	return min_sample_index;
}

static int getNewSampleIndex(std::mutex *mu, int *sample_index_global, int sample_offset, int tid, std::vector<int> *running_index_list)
{
	int sample_index;

	mu->lock();
	sample_index = *sample_index_global + sample_offset;
	*sample_index_global = *sample_index_global + 1;
	(*running_index_list)[tid] = sample_index;
	mu->unlock();

	return sample_index;
}

void doPreProcessing(void *d) {
	PreProcessingThreadData *data = (PreProcessingThreadData *)d;
	ConfigData *config_data = data->config_data;
	std::vector<long> *latency = data->latency;
	int instance_id = data->instance_id;
	int tid = data->tid;
	IInferenceApplication *app = data->app;
	int *sample_index_global = data->sample_index;
	std::mutex *mu = data->mu;
	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int batch = config_data->instances.at(instance_id).batch;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int sample_index = 0;
	int index = 0;
	long stuckWhile = 0;
	int batch_thread_num = config_data->instances.at(instance_id).batch_thread_num;
	std::vector<int> *cur_running_index_list = data->cur_running_index;

	sample_index = getNewSampleIndex(mu, sample_index_global, sample_offset, tid, cur_running_index_list);

	index = sample_index * batch;

	while((sample_size == 0 || (sample_size > 0 && sample_index < sample_offset + sample_size)) && exit_flag == false) {
		int buffer_index = sample_index % buffer_num;	
		bool is_runnable = data->model->isPreprocessingRunnable(buffer_index);

		while((!is_runnable || sample_index >= getMinSampleIndex(cur_running_index_list, sample_index) + buffer_num) && exit_flag == false) {
			usleep(SLEEP_TIME);
			stuckWhile++;
			is_runnable = data->model->isPreprocessingRunnable(buffer_index);
		}

		(*latency)[sample_index - sample_offset] = getTime();
		readData(tid, data->model->net_input_buffers[buffer_index][0], app, data->model->total_input_size, batch, batch_thread_num, index);
		data->model->updateInputSignals(buffer_index, true);	

		sample_index = getNewSampleIndex(mu, sample_index_global, sample_offset, tid, cur_running_index_list);
		index = sample_index * batch;
	}

	fprintf(stderr, "stuckWhile(front thread: %d): %ld\n", tid, stuckWhile);
}


void doPostProcessing(void *d) {
	PostProcessingThreadData *data = (PostProcessingThreadData *)d;
	ConfigData *config_data = data->config_data;
	std::vector<long> *latency = data->latency;
	int instance_id = data->instance_id;
	int tid = data->tid;
	IInferenceApplication *app = data->app;
	int instance_num = config_data->instance_num;
	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int batch = config_data->instances.at(instance_id).batch;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	std::string network_name = config_data->instances.at(instance_id).network_name;
	int sample_index = sample_offset + tid;
	int buffer_id = 0;
	long stuckWhile = 0;
	int *sample_index_global = data->sample_index;
	std::mutex *mu = data->mu;
	std::vector<int> *cur_running_index_list = data->cur_running_index;
	float **output_pointers;

	output_pointers = (float **) calloc(data->model->network_output_number, sizeof(float *));

	sample_index = getNewSampleIndex(mu, sample_index_global, sample_offset, tid, cur_running_index_list);

	while((sample_size == 0 || (sample_size > 0 && sample_index < sample_offset + sample_size)) && exit_flag == false) {
		int buffer_index = sample_index % buffer_num;	
		bool is_runnable = data->model->isPostprocessingRunnable(buffer_index);

		while((!is_runnable || sample_index >= getMinSampleIndex(cur_running_index_list, sample_index) + buffer_num) && exit_flag == false) {
			usleep(SLEEP_TIME);	
			stuckWhile++;
			is_runnable = data->model->isPostprocessingRunnable(buffer_index);
		}

		buffer_id = sample_index % buffer_num;

		for(int iter = 0 ; iter < data->model->network_output_number; iter++) {
			output_pointers[iter] = data->model->net_output_buffers[buffer_id][iter];
		}

		app->postprocessing1(tid, sample_index, output_pointers, data->model->network_output_number, batch);

		data->model->updateOutputSignals(buffer_index, false);	

		app->postprocessing2(tid, sample_index, batch);

		if(tid == 0 && instance_id == 0) {
			std::cerr<<"[TEST | "<<(sample_index+1)*instance_num<<" / "<<sample_size*instance_num<<"]\r";	
		}

		(*latency)[sample_index - sample_offset] = getTime() - (*latency)[sample_index - sample_offset];

		sample_index = getNewSampleIndex(mu, sample_index_global, sample_offset, tid, cur_running_index_list);
	}

	fprintf(stderr, "stuckWhile(back thread: %d): %ld\n", tid, stuckWhile);

	free(output_pointers);
}

void doInference(void *d) {
	InferenceThreadData *data = (InferenceThreadData *)d;
	ConfigData *config_data = data->config_data;
	int instance_id = data->instance_id;
	int device_id = data->tid;
	Model *model = data->model;

	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int stream_num = config_data->instances.at(instance_id).stream_numbers.at(device_id);
	std::string network_name = config_data->instances.at(instance_id).network_name;
	int sample_index = sample_offset;
	std::vector<int> ready(stream_num, 1);
	std::vector<int> assignedSampleId(stream_num, -1);
	int sleep_time = 0;
	long stuckWhile = 0;
	int next_buffer_index = 0;
	int assigned_buffer_id = 0;
	int next_stream_index = 0;
	int min_sample_index = 0;
	std::vector<int> stream_balance(stream_num, 0);
	std::list<int> available_streams;

	for(int iter = 0; iter < stream_num ; iter++) {
		available_streams.push_back(iter);
	}

	while((sample_size == 0 || (sample_size > 0 && sample_index < sample_offset + sample_size)) && exit_flag == false) {
		while(exit_flag == false) {
			int buffer_index = sample_index % buffer_num;
			bool is_runnable = model->stages[device_id]->isRunnable(buffer_index);

			if(is_runnable && sample_index < min_sample_index + buffer_num) {
				if(available_streams.size() > 0) {
					next_buffer_index = sample_index % buffer_num;
					next_stream_index = available_streams.front();
					available_streams.pop_front();
					break;
				}
			}
			min_sample_index = sample_offset + sample_size;
			for(int iter = 0; iter < stream_num; iter++) {
				if(ready[iter] == 0) {
					if(model->checkInferenceDone(device_id, iter)) {
						assigned_buffer_id = assignedSampleId[iter] % buffer_num;

						model->stages[device_id]->updateInputSignals(assigned_buffer_id, false);
						model->stages[device_id]->updateOutputSignals(assigned_buffer_id, true);
						ready[iter] = 1;
						assignedSampleId[iter] = -1;
						available_streams.push_back(iter);
					}
					if(min_sample_index > assignedSampleId[iter] && assignedSampleId[iter] >= 0) {
						min_sample_index = assignedSampleId[iter];	
					}

				}
			}

			if(available_streams.size() > 0) {
				stuckWhile++;
			}
			usleep(SLEEP_TIME);
			sleep_time++;


			if(sleep_time > MAX_TIMEOUT) {
				exit_flag = true;
				printf("timeout is reached. program will be terminated.\n");
			}
		}	
		
		sleep_time = 0;
		assignedSampleId[next_stream_index] = sample_index;
		model->infer(device_id, next_stream_index, next_buffer_index);
		stream_balance[next_stream_index]++;
		ready[next_stream_index] = 0;

		sample_index++;
	}

	for(int iter = 0; iter < stream_num; iter++) {
		if(ready[iter] == 0) {
			if(exit_flag == false) 
			{
				model->waitUntilInferenceDone(device_id, iter);
			}
			assigned_buffer_id = assignedSampleId[iter] % buffer_num;
			model->stages[device_id]->updateInputSignals(assigned_buffer_id, false);
			model->stages[device_id]->updateOutputSignals(assigned_buffer_id, true);
			ready[iter] = 1;
		}	
		fprintf(stderr, "device id: %d, stream_id: %d, executed_num: %d\n", device_id, iter, stream_balance[iter]);
	}

	fprintf(stderr, "stuckWhile(device id: %d): %ld\n", device_id, stuckWhile);
}
