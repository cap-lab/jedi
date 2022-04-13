#include "runner.h"

int stickThisThreadToCore(int core_id) {
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

	stickThisThreadToCore(runner->pre_thread_core);

	while (!runner->exit_flag) {
		while(runner->signals.at(PRE_LOCK) && !runner->exit_flag) {
			usleep(SLEEP_TIME);	
		}
		if(runner->exit_flag) break;

		runner->setInputData();
		runner->signals.at(PRE_LOCK) = 1;
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

	stickThisThreadToCore(runner->post_thread_core);

	while(!runner->exit_flag) {
		while(runner->signals.at(POST_LOCK) && !runner->exit_flag) {
			usleep(SLEEP_TIME);	
		}
		if(runner->exit_flag) break;

		runner->postProcess();	
		runner->signals.at(POST_LOCK) = 1;
	}
}


Runner::Runner(std::string config_file_name) {
	this->config_data = new ConfigData(config_file_name, this->apps);
	this->device_num = this->config_data->instances.at(0).device_num;
}


Runner::~Runner() {
	if (!(this->models.empty())) {
		wrapup();	
	}
	delete this->config_data;
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


void Runner::set_thread_cores(int pre_thread_core, int post_thread_core) {
	this->pre_thread_core = pre_thread_core;
	this->post_thread_core = post_thread_core;
}


void Runner::setInputData() {
	Model *model = this->models.at(0);
	int input_size = this->width * this->height * this->channel * sizeof(float);

	memcpy(model->input_buffers.at(0), this->image_data, input_size);	
}


void Runner::postProcess() {
	float **output_pointers;
	Model *model = this->models.at(0);
	IInferenceApplication *app = this->apps.at(0);

	output_pointers = (float **) calloc(model->network_output_number, sizeof(float *));

	for(int iter = 0 ; iter < model->network_output_number; iter++) {
		output_pointers[iter] = model->output_buffers[iter];
	}

	((YoloApplication *)app)->postprocessing2(this->width, this->height, output_pointers, model->network_output_number);
}


void Runner::runThreads() {
	for (int iter = 0; iter < LOCK_NUM; iter++) {
		this->signals.push_back(1);
	}

	this->threads.push_back(std::thread(runPreProcess, this));	
	this->threads.push_back(std::thread(runInference, this));	
	this->threads.push_back(std::thread(runPostProcess, this));
}


void Runner::run_with_data(char *data, int width, int height, int channel) {
	this->width = width;
	this->height = height;
	this->channel = channel;
	this->image_data = data;

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
