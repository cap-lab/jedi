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
#include "util.h"

#include "inference_application.h"

#define MAX_TIMEOUT (100000)

static long getAverageLatency(int instance_id, ConfigData *config_data, std::vector<long> latency)
{
	long sum  = 0;
	int nSize = latency.size(); 

	for(int iter = 0 ; iter < nSize ; iter++) {
		sum += latency[iter];
	}

	return sum / (long) nSize;
}

void printAverageLatency(int instance_id, ConfigData *config_data, std::vector<std::vector<long>> latencies, std::ofstream &fp) {
	int latency_type_size = latencies.size();
	//stage_profile_file_name
	for(int iter = 0 ; iter < latency_type_size ; iter++) {
		long average_latency = getAverageLatency(iter, config_data, latencies[iter]);
		fp << average_latency << std::endl;
		if (iter == latency_type_size - 1) {
			std::cout<< "end2end average latency ("<< instance_id <<"): " << average_latency << std::endl;
		} else {
			std::cout<< "average latency ("<< instance_id <<", stage: "<<iter<<"): " << average_latency << std::endl;
		}
	}
}

static void readData(int thread_id, int input_tensor_index, const char *input_name, void *input_buffer, IInferenceApplication *app, int input_tensor_size, int batch, int batch_thread_num, int index)
{
	int input_size = input_tensor_size / batch;
	#pragma omp parallel num_threads(batch_thread_num)
	#pragma omp for
	for(int iter = 0; iter < batch; iter++) {
		app->preprocessing(thread_id, input_tensor_index, input_name, index, iter, (float *) input_buffer + iter * input_size);
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

void doInferenceAll(ConfigData &config_data, IInferenceApplication *app, Model *model, int instance_id, std::vector<long> *latency)
{
	int sample_index = 0;
	int sample_offset = config_data.instances.at(instance_id).offset;
	int sample_size = config_data.instances.at(instance_id).sample_size;
	int batch = config_data.instances.at(instance_id).batch;
	int index = sample_index * batch;
	int device_num = config_data.instances.at(instance_id).device_num;
	void **output_pointers;
	void (Model::*funcPtr[device_num])(int, int, int);

	for (int iter = 0; iter < device_num; iter++) {
		if (config_data.instances.at(instance_id).devices.at(iter) == DEVICE_GPU &&
		config_data.instances.at(instance_id).cuda_graphs.at(iter) == true) {
			funcPtr[iter] = &Model::graphLaunch;
		} else {
			funcPtr[iter] = &Model::infer;
		}
	}

	

	output_pointers = (void **)calloc(model->network_output_number, sizeof(void *));

	while (sample_index < sample_offset + sample_size)
	{
		index = sample_index * batch;

		auto input_size_vec = model->stages[0]->input_size_vec;
		int input_tensor_index = 0;

		(*latency)[sample_index - sample_offset] = getTime();

		for (auto iter = input_size_vec.begin(); iter != input_size_vec.end(); iter++)
		{
			int input_size = 1;
			nvinfer1::Dims dims = iter->second;

			for (int iter2 = 0; iter2 < dims.nbDims; iter2++)
				input_size = input_size * dims.d[iter2];

			// input tensor index is needed
			readData(0, input_tensor_index, iter->first.c_str(), model->net_input_buffers[0][input_tensor_index], app, input_size, batch, 1, index);
			input_tensor_index++;
		}

		for (int iter = 0; iter < device_num; iter++)
		{
			(model->*funcPtr[iter])(iter, 0, 0);
			model->waitUntilInferenceDone(iter, 0);
		}

		for (int iter = 0; iter < model->network_output_number; iter++)
		{
			output_pointers[iter] = model->net_output_buffers[0][iter];
		}

		// no use signals (tmp1) in this version
		app->postprocessing1(0, sample_index, output_pointers, model->network_output_number, batch);
		app->postprocessing2(0, sample_index, batch);

		(*latency)[sample_index - sample_offset] = getTime() - (*latency)[sample_index - sample_offset];

		std::cerr << "[TEST | " << (sample_index + 1) << " / " << sample_size << "]\r";

		sample_index += 1;
	}

	free(output_pointers);
}

void doPreProcessing(void *d) {
	PreProcessingThreadData *data = (PreProcessingThreadData *)d;
	ConfigData *config_data = data->config_data;
	std::vector<std::vector<long>> *latency = data->latency;
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
	int total_latency_index = (*latency).size() - 1;
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

		auto input_size_vec = data->model->stages[0]->input_size_vec;
		int input_tensor_index = 0;

		(*latency)[total_latency_index][sample_index - sample_offset] = getTime();
#ifndef DISABLE_PROFILE
		(*latency)[0][sample_index - sample_offset] = (*latency)[total_latency_index][sample_index - sample_offset];
#endif
		for(auto iter = input_size_vec.begin(); iter != input_size_vec.end(); iter++) {
			int input_size = 1;
			nvinfer1::Dims dims = iter->second;

			for(int iter2 = 0; iter2 < dims.nbDims; iter2++)
				input_size = input_size * dims.d[iter2];

			// input tensor index is needed
			readData(tid, input_tensor_index, iter->first.c_str(), data->model->net_input_buffers[buffer_index][input_tensor_index], app, input_size, batch, batch_thread_num, index);
			input_tensor_index++;
		}
#ifndef DISABLE_PROFILE
		(*latency)[0][sample_index - sample_offset] = getTime() - (*latency)[0][sample_index - sample_offset];
#endif
		data->model->updateInputSignals(buffer_index, true);

		sample_index = getNewSampleIndex(mu, sample_index_global, sample_offset, tid, cur_running_index_list);
		index = sample_index * batch;
	}

	fprintf(stderr, "stuckWhile(front thread: %d): %ld\n", tid, stuckWhile);
}


void doPostProcessing(void *d) {
	PostProcessingThreadData *data = (PostProcessingThreadData *)d;
	ConfigData *config_data = data->config_data;
	std::vector<std::vector<long>> *latency = data->latency;
	int instance_id = data->instance_id;
	int tid = data->tid;
	IInferenceApplication *app = data->app;
	int instance_num = config_data->instance_num;
	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int batch = config_data->instances.at(instance_id).batch;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int sample_index = sample_offset + tid;
	int buffer_id = 0;
	long stuckWhile = 0;
	int *sample_index_global = data->sample_index;
	std::mutex *mu = data->mu;
	std::vector<int> *cur_running_index_list = data->cur_running_index;
	int total_latency_index = (*latency).size() - 1;
	void **output_pointers;

	output_pointers = (void **) calloc(data->model->network_output_number, sizeof(void *));

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
#ifndef DISABLE_PROFILE
		(*latency)[total_latency_index-1][sample_index - sample_offset] = getTime();
#endif
		app->postprocessing1(tid, sample_index, output_pointers, data->model->network_output_number, batch);

		data->model->updateOutputSignals(buffer_index, false);

		app->postprocessing2(tid, sample_index, batch);

		long end_time = getTime();
		(*latency)[total_latency_index][sample_index - sample_offset] = end_time - (*latency)[total_latency_index][sample_index - sample_offset];
#ifndef DISABLE_PROFILE
		(*latency)[total_latency_index-1][sample_index - sample_offset] = end_time - (*latency)[total_latency_index-1][sample_index - sample_offset];
#endif

		if(tid == 0 && instance_id == 0) {
			std::cerr<<"[TEST | "<<(sample_index+1)*instance_num<<" / "<<sample_size*instance_num<<"]\r";	
		}

		sample_index = getNewSampleIndex(mu, sample_index_global, sample_offset, tid, cur_running_index_list);
	}

	fprintf(stderr, "stuckWhile(back thread: %d): %ld\n", tid, stuckWhile);

	free(output_pointers);
}


#ifdef STRING_PER_BUFFER
void doInferenceGraph(void *d) {
	InferenceThreadData *data = (InferenceThreadData *)d;
	ConfigData *config_data = data->config_data;
	int instance_id = data->instance_id;
	std::vector<std::vector<long>> *latency = data->latency;
	int device_id = data->tid;
	Model *model = data->model;

	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int stream_num = config_data->instances.at(instance_id).stream_numbers.at(device_id);
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
	std::vector<bool> stream_available(stream_num, true);
	int available_stream_num = stream_num;

	while((sample_size == 0 || (sample_size > 0 && sample_index < sample_offset + sample_size)) && exit_flag == false) {
		while(exit_flag == false) {
			int buffer_index = sample_index % buffer_num;
			bool is_runnable = model->stages[device_id]->isRunnable(buffer_index);
			int stream_index = buffer_index % stream_num;

			if(is_runnable && sample_index < min_sample_index + buffer_num && stream_available[stream_index] == true) {
				next_buffer_index = sample_index % buffer_num;
				next_stream_index = stream_index;
				stream_available[stream_index] = false;
				available_stream_num--;
				break;
			}
			min_sample_index = sample_offset + sample_size;
			for(int iter = 0; iter < stream_num; iter++) {
				if(ready[iter] == 0) {
					if(model->checkInferenceDone(device_id, iter)) {
#ifndef DISABLE_PROFILE
						(*latency)[device_id+1][assignedSampleId[iter] - sample_offset] = getTime() - (*latency)[device_id+1][assignedSampleId[iter] - sample_offset];
#endif
						assigned_buffer_id = assignedSampleId[iter] % buffer_num;

						model->stages[device_id]->updateInputSignals(assigned_buffer_id, false);
						model->stages[device_id]->updateOutputSignals(assigned_buffer_id, true);
						ready[iter] = 1;
						assignedSampleId[iter] = -1;
						stream_available[iter] = true;
						available_stream_num++;
					}
					if(min_sample_index > assignedSampleId[iter] && assignedSampleId[iter] >= 0) {
						min_sample_index = assignedSampleId[iter];
					}
				}
			}

			if(available_stream_num > 0) {
				stuckWhile++;
			}
			usleep(SLEEP_TIME);
			sleep_time++;

			if(sleep_time > MAX_TIMEOUT) {
				if(sleep_time > MAX_TIMEOUT * 10) {
					printf("timeout is reached. program will be terminated.\n");
					exit_flag = true;
				}
				if((sleep_time % MAX_TIMEOUT) == 1) {
					printf("timeout check.\n");
				}
			}
		}

		sleep_time = 0;
		assignedSampleId[next_stream_index] = sample_index;
#ifndef DISABLE_PROFILE
		(*latency)[device_id+1][sample_index - sample_offset] = getTime();
#endif
		model->graphLaunch(device_id, next_stream_index, next_buffer_index);
		stream_balance[next_stream_index]++;
		ready[next_stream_index] = 0;

		sample_index++;
	}

	for(int iter = 0; iter < stream_num; iter++) {
		if(ready[iter] == 0) {
			if(exit_flag == false)
			{
				model->waitUntilInferenceDone(device_id, iter);
#ifndef DISABLE_PROFILE
				(*latency)[device_id+1][assignedSampleId[iter] - sample_offset] = getTime() - (*latency)[device_id+1][assignedSampleId[iter] - sample_offset];
#endif
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
#endif

void doInference(void *d) {
	InferenceThreadData *data = (InferenceThreadData *)d;
	ConfigData *config_data = data->config_data;
	int instance_id = data->instance_id;
	std::vector<std::vector<long>> *latency = data->latency;
	int device_id = data->tid;
	Model *model = data->model;
	void (Model::*funcPtr)(int, int, int);

	int sample_offset = config_data->instances.at(instance_id).offset;
	int sample_size = config_data->instances.at(instance_id).sample_size;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int stream_num = config_data->instances.at(instance_id).stream_numbers.at(device_id);
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

	if (config_data->instances.at(instance_id).devices.at(device_id) == DEVICE_GPU &&
		config_data->instances.at(instance_id).cuda_graphs.at(device_id) == true) {
		funcPtr = &Model::graphLaunch;
	} else {
		funcPtr = &Model::infer;
	}

	while((sample_size == 0 || (sample_size > 0 && sample_index < sample_offset + sample_size)) && exit_flag == false) {
		while(exit_flag == false) {
			int buffer_index = sample_index % buffer_num;
			bool is_runnable = model->stages[device_id]->isRunnable(buffer_index);

			if(is_runnable && sample_index < min_sample_index + buffer_num && available_streams.size() > 0) {
				next_buffer_index = sample_index % buffer_num;
				next_stream_index = available_streams.front();
				available_streams.pop_front();
				break;
			}
			min_sample_index = sample_offset + sample_size;
			for(int iter = 0; iter < stream_num; iter++) {
				if(ready[iter] == 0) {
					if(model->checkInferenceDone(device_id, iter)) {
#ifndef DISABLE_PROFILE
						(*latency)[device_id+1][assignedSampleId[iter] - sample_offset] = getTime() - (*latency)[device_id+1][assignedSampleId[iter] - sample_offset];
#endif
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
				if(sleep_time > MAX_TIMEOUT * 10) {
					printf("timeout is reached. program will be terminated.\n");
					exit_flag = true;
				}
				if((sleep_time % MAX_TIMEOUT) == 1) {
					printf("timeout check.\n");
				}
			}
		}	
		
		sleep_time = 0;
		assignedSampleId[next_stream_index] = sample_index;
#ifndef DISABLE_PROFILE
		(*latency)[device_id+1][sample_index - sample_offset] = getTime();
#endif
		(model->*funcPtr)(device_id, next_stream_index, next_buffer_index);
		stream_balance[next_stream_index]++;
		ready[next_stream_index] = 0;
		sample_index++;
	}

	for(int iter = 0; iter < stream_num; iter++) {
		if(ready[iter] == 0) {
			if(exit_flag == false) 
			{
				model->waitUntilInferenceDone(device_id, iter);
#ifndef DISABLE_PROFILE
				(*latency)[device_id+1][assignedSampleId[iter] - sample_offset] = getTime() - (*latency)[device_id+1][assignedSampleId[iter] - sample_offset];
#endif
			}
			assigned_buffer_id = assignedSampleId[iter] % buffer_num;
			model->stages[device_id]->updateInputSignals(assigned_buffer_id, false);

			/*int binding_num = model->stages[device_id]->contexts[0]->getEngine().getNbIOTensors();
			for(int iter1 = 0; iter1 < binding_num; iter1++) {
				auto const& name = model->stages[device_id]->contexts[0]->getEngine().getIOTensorName(iter1);
				auto const& mode = model->stages[device_id]->contexts[0]->getEngine().getTensorIOMode(name);
				bool isInput = (mode == nvinfer1::TensorIOMode::kINPUT) ? true : false;

				if(!isInput) {
					TensorAllocator *allocator = (TensorAllocator *)model->stages[device_id]->contexts[0]->getOutputAllocator(name);
					char *data = (char *) allocator->getBuf();
				    std::ofstream p("merong.data", std::ios::binary);
				    p.write(data, allocator->getSize());
					break;
				}
			}*/

			model->stages[device_id]->updateOutputSignals(assigned_buffer_id, true);
			ready[iter] = 1;
		}	
		fprintf(stderr, "device id: %d, stream_id: %d, executed_num: %d\n", device_id, iter, stream_balance[iter]);
	}
	fprintf(stderr, "stuckWhile(device id: %d): %ld\n", device_id, stuckWhile);
}
