#include "runner2.h"
#include "image_opencv.h"
#include <iostream>

extern std::vector<long> pre_time_vec, post_time_vec;

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

void Runner2::initializePreAndPostprocessing() {
	std::string network_name = config_data->instances.at(0).network_name;
	int batch_size = config_data->instances.at(0).batch;
	int pre_thread_num = config_data->instances.at(0).pre_thread_num;
	int post_thread_num = config_data->instances.at(0).post_thread_num;

	apps[0]->initializePreprocessing(network_name, batch_size, pre_thread_num);
	apps[0]->initializePostprocessing(network_name, batch_size, post_thread_num);
}


Runner2::Runner2(std::string config_file_name, int width, int height, int channel, int step) {
	this->config_data = new ConfigData(config_file_name, this->apps);
	this->device_num = this->config_data->instances.at(0).device_num;
	this->width = width;
	this->height = height;
	this->channel = channel;
	this->step = step;
	this->input_size =  width * height * channel;
}


Runner2::~Runner2() {
	if (!(this->models.empty())) {
		wrapup();	
	}
	delete this->config_data;
}


void Runner2::generateModels() {
	Model *model = new Model(config_data, 0, this->apps.at(0));

	model->initializeModel();
	model->initializeBuffers();

	this->models.emplace_back(model);
}


void Runner2::init() {
	this->exit_flag = false;
	cudaSetDeviceFlags(cudaDeviceMapHost);

	generateModels();
	initializePreAndPostprocessing();

	input_buffer = new float[width * height * channel];
	output_pointers = (float **) calloc(this->models.at(0)->network_output_number, sizeof(float *));
}


void Runner2::loadImage(char *filename, int w, int h, int c, int *orig_width, int *orig_height, float *input) {
	loadImageResize(filename, w, h, c, orig_width, orig_height, input);
}


void Runner2::setInputData(float *input_buffer_) {
	Model *model = this->models.at(0);
	long curr_time = getTime();

	memcpy(model->input_buffers.at(0), input_buffer_, input_size*sizeof(float));	

	pre_time_vec.push_back(getTime() - curr_time);
}


void Runner2::setInputData2(char *image_data) {
	Model *model = this->models.at(0);
	int w = this->width, h = this->height, c = this->channel;
	long curr_time = getTime();

	for (int y = 0; y < h; ++y) {
		for (int k = 0; k < c; ++k) {
			for (int x = 0; x < w; ++x) {
				input_buffer[k*w*h + y*w + x] = image_data[y*step + x*c + k] / 255.0f;
			}
		}
	}

	memcpy(model->input_buffers.at(0), input_buffer, input_size*sizeof(float));	

	pre_time_vec.push_back(getTime() - curr_time);
}


void Runner2::runInference() {
	Model *model = this->models.at(0);
	int device_num = this->device_num;

	for(int device_id = 0; device_id < device_num; device_id++) {
		model->infer(device_id, 0, 0);
	}
}


void Runner2::getOutputData(float **output_buffers) {
	Model *model = this->models.at(0);

	for(int iter = 0 ; iter < model->network_output_number; iter++) {
		output_buffers[iter] = model->output_buffers[iter];
	}
}


void Runner2::doPostProcessing(char *input_data, char *result_file_name) {
	Model *model = this->models.at(0);
	IInferenceApplication *app = this->apps.at(0);
	long curr_time = getTime();

	for(int iter = 0 ; iter < model->network_output_number; iter++) {
		this->output_pointers[iter] = model->output_buffers[iter];
	}

	((YoloApplication *)app)->postprocessing2(this->width, this->height, this->step, input_data, this->output_pointers, model->network_output_number, result_file_name);

	post_time_vec.push_back(getTime() - curr_time);
}


void Runner2::doPostProcessing(float **output_pointers_, char *input_data, char *result_file_name) {
	Model *model = this->models.at(0);
	IInferenceApplication *app = this->apps.at(0);
	long curr_time = getTime();

	((YoloApplication *)app)->postprocessing2(this->width, this->height, this->step, input_data, output_pointers_, model->network_output_number, result_file_name);

	post_time_vec.push_back(getTime() - curr_time);
}

void Runner2::setThreadCores(int core_id) {
	stickThisThreadToCore(core_id);
}

void Runner2::saveProfileResults(char *max_filename, char *avg_filename, char *min_filename) {
	Model *model = models.at(0);

	model->printProfile(max_filename, avg_filename, min_filename);
}

void Runner2::saveResults(char *result_file_name) {
	IInferenceApplication *app = this->apps.at(0);
	std::string file_name(result_file_name);

	((YoloApplication *)app)->saveResults(file_name);
}

void Runner2::finalizeData() {
	Model *model = this->models.at(0);
	IInferenceApplication *app = this->apps.at(0);

	model->finalizeBuffers();
	model->finalizeModel();

	delete app;
	delete model;

	models.clear();
	apps.clear();
}


void Runner2::wrapup() {
	exit_flag = true;

	finalizeData();
}
