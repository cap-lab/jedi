#include "runner.h"
#include <iostream>

std::vector<long> pre_time_vec, post_time_vec;

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

void runPreProcess(void *d) {
	Runner *runner = (Runner *)d;
	long curr_time = getTime();

	stickThisThreadToCore(runner->pre_thread_core);

	while (!runner->exit_flag) {
		while(runner->signals.at(PRE_LOCK) && !runner->exit_flag) {
			usleep(SLEEP_TIME);	
		}
		if(runner->exit_flag) break;

		curr_time = getTime();
		runner->setInputData();
		runner->signals.at(PRE_LOCK) = 1;
		pre_time_vec.push_back(getTime() - curr_time);
	}
}

void runInference(void *d) {
	Runner *runner = (Runner *)d;
	Model *model = runner->models.at(0);
	int device_num = runner->device_num;

	while(!runner->exit_flag) {
		while(runner->signals.at(INFER_LOCK) && !runner->exit_flag) {
			usleep(SLEEP_TIME);	
		}
		if(runner->exit_flag) break;

		for(int device_id = 0; device_id < device_num; device_id++) {
			model->infer(device_id, 0, 0);
		}
		runner->signals.at(INFER_LOCK) = 1;
	}
}


void runPostProcess(void *d) {
	Runner *runner = (Runner *)d;
	long curr_time = getTime();

	stickThisThreadToCore(runner->post_thread_core);

	while(!runner->exit_flag) {
		while(runner->signals.at(POST_LOCK) && !runner->exit_flag) {
			usleep(SLEEP_TIME);	
		}
		if(runner->exit_flag) break;

		curr_time = getTime();
		runner->postProcess();	
		runner->signals.at(POST_LOCK) = 1;
		post_time_vec.push_back(getTime() - curr_time);
	}
}


Runner::Runner(std::string config_file_name, int width, int height, int channel, int step) {
	this->config_data = new ConfigData(config_file_name, this->apps);
	this->device_num = this->config_data->instances.at(0).device_num;
	this->width = width;
	this->height = height;
	this->channel = channel;
	this->step = step;
	this->input_buffer = new float[width * height * channel];
	this->image_data = new char[width * height * channel];
}


Runner::~Runner() {
	if (!(this->models.empty())) {
		wrapup();	
	}
	delete this->config_data;
	delete this->input_buffer;
}


void Runner::generateModels() {
	Model *model = new Model(config_data, 0, this->apps.at(0));

	model->initializeModel();
	model->initializeBuffers();

	this->models.emplace_back(model);
}


void Runner::initializePreAndPostprocessing() {
	std::string network_name = config_data->instances.at(0).network_name;
	int batch_size = config_data->instances.at(0).batch;
	int pre_thread_num = config_data->instances.at(0).pre_thread_num;
	int post_thread_num = config_data->instances.at(0).post_thread_num;

	apps[0]->initializePreprocessing(network_name, batch_size, pre_thread_num);
	apps[0]->initializePostprocessing(network_name, batch_size, post_thread_num);
}


void Runner::init() {
	this->exit_flag = false;
	cudaSetDeviceFlags(cudaDeviceMapHost);

	generateModels();
	initializePreAndPostprocessing();
	runThreads();
}


void Runner::setThreadCores(int pre_thread_core, int post_thread_core) {
	this->pre_thread_core = pre_thread_core;
	this->post_thread_core = post_thread_core;
}


void Runner::setInputData() {
	Model *model = this->models.at(0);
	int w = this->width;
	int h = this->height;
	int c = this->channel;
	int step = this->step;
	int input_size =  w * h * c * sizeof(float);

	for (int y = 0; y < h; ++y) {
		for (int k = 0; k < c; ++k) {
			for (int x = 0; x < w; ++x) {
				this->input_buffer[k*w*h + y*w + x] = this->image_data[y*step + x*c + k] / 255.0f;
			}
		}
	}
	memcpy(model->input_buffers.at(0), this->input_buffer, input_size);	
}


void Runner::postProcess() {
	float **output_pointers;
	Model *model = this->models.at(0);
	IInferenceApplication *app = this->apps.at(0);

	output_pointers = (float **) calloc(model->network_output_number, sizeof(float *));

	for(int iter = 0 ; iter < model->network_output_number; iter++) {
		output_pointers[iter] = model->output_buffers[iter];
	}

	((YoloApplication *)app)->postprocessing2(this->width, this->height, this->step, this->image_data, output_pointers, model->network_output_number, this->result_file_name);
	this->result_file_name = nullptr;
}


void Runner::runThreads() {
	for (int iter = 0; iter < LOCK_NUM; iter++) {
		this->signals.push_back(1);
	}

	this->threads.push_back(std::thread(runPreProcess, this));	
	this->threads.push_back(std::thread(runInference, this));	
	this->threads.push_back(std::thread(runPostProcess, this));
}


void Runner::run(char *data) {
	int input_size = this->width * this->height * this->channel * sizeof(char);
	memcpy(this->image_data, data, input_size);	

	signals.at(PRE_LOCK) = 0;
	while(!signals.at(PRE_LOCK)) {
		usleep(SLEEP_TIME);	
	}

	signals.at(INFER_LOCK) = 0;
	while(!signals.at(INFER_LOCK)) {
		usleep(SLEEP_TIME);	
	}

	signals.at(POST_LOCK) = 0;
	while(!signals.at(POST_LOCK)) {
		usleep(SLEEP_TIME);	
	}
}


void Runner::run(char *data, char *result_file_name) {
	this->result_file_name = result_file_name;
	run(data);
}


void Runner::saveProfileResults(char *max_filename, char *avg_filename, char *min_filename) {
	Model *model = models.at(0);

	model->printProfile(max_filename, avg_filename, min_filename);
}


void Runner::finalizeData() {
	Model *model = this->models.at(0);
	IInferenceApplication *app = this->apps.at(0);

	model->finalizeBuffers();
	model->finalizeModel();

	delete app;
	delete model;

	models.clear();
	apps.clear();
}


void Runner::wrapup() {
	exit_flag = true;

	for (int iter = 0; iter < LOCK_NUM; iter++) {
		this->threads.at(iter).join();	
	}

	finalizeData();
}
