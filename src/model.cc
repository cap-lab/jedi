#include <iostream>
#include <vector>
#include <cassert>
#include <cctype>

#include <NvInfer.h>
#include <tkDNN/tkdnn.h>
#include <tkDNN/DarknetParser.h>

#include "cuda.h"

#include "model.h"
#include "variable.h"


typedef struct _CalibrationTable {
	std::string key;
	std::string value;
} CalibrationTable;


static long getTime() {
	struct timespec time;
	if(0 != clock_gettime(CLOCK_REALTIME, &time)) {
		std::cerr<<"Something wrong on clock_gettime()"<<std::endl;		
		exit(-1);
	}
	return (time.tv_nsec) / 1000 + time.tv_sec * 1000000; // us
}

static bool fileExist(std::string fname) {
    std::ifstream dataFile (fname.c_str(), std::ios::in | std::ios::binary);
    if(!dataFile)
    	return false;
    return true;
}


Model::Model(ConfigData *config_data, int instance_id, IInferenceApplication *app) {
	this->config_data = config_data;
	this->instance_id = instance_id;
	this->network_output_number = 0;
	this->app = app;

	int device_num = config_data->instances.at(instance_id).device_num;
	for(int iter1 = 0; iter1 < device_num; iter1++) {
		netRTs.push_back(std::vector<tk::dnn::NetworkRT *>());
	}

	for(int iter=0; iter<2; iter++) {
		std::vector<long> vec;	
		dla_profile_vec.push_back(vec);
	}
}

Model::~Model() {
	start_bindings.clear();
	binding_size.clear();
	is_net_output.clear();
	stream_buffers.clear();
	input_buffers.clear();
	output_buffers.clear();

	for(unsigned int iter1 = 0; iter1 < netRTs.size(); iter1++) {
		netRTs[iter1].clear();
	}
	netRTs.clear();

	for(unsigned int iter1 = 0; iter1 < contexts.size(); iter1++) {
		contexts.at(iter1).clear();	
	}

	for(unsigned int iter1 = 0; iter1 < streams.size(); iter1++) {
		streams.at(iter1).clear();
	}
}

void Model::getModelFileName(int curr, std::string &plan_file_name) {
	std::string model_dir = config_data->instances.at(instance_id).model_dir;
	std::string cut_points_name;
	std::string device_name;
	std::string data_type_name;
	std::string image_size_name;
	int device = config_data->instances.at(instance_id).devices.at(curr);
	int batch = config_data->instances.at(instance_id).batch;
	int data_type = config_data->instances.at(instance_id).data_type;
	int prev_cut_point = 0, curr_cut_point = 0;
	
	if(curr > 0) {
		prev_cut_point = config_data->instances.at(instance_id).cut_points.at(curr-1) + 1;
	}
	curr_cut_point = config_data->instances.at(instance_id).cut_points.at(curr);

	cut_points_name = std::to_string(prev_cut_point) + "." + std::to_string(curr_cut_point);

	if(device == DEVICE_DLA) {
		device_name = "DLA";
	}
	else {
		device_name = "GPU";
	}

	if(data_type == TYPE_FP32) {
		data_type_name = "FP32";
	}
	else if(data_type == TYPE_FP16) {
		data_type_name = "FP16";
	}
	else if(data_type == TYPE_INT8) {
		data_type_name = "INT8";
	}

	image_size_name = std::to_string(input_dim.width) + "x" + std::to_string(input_dim.height);

	plan_file_name = model_dir + "/model" + image_size_name + "_" + cut_points_name + "_" + device_name + "_" + data_type_name + "_" + std::to_string(batch) + ".rt";
	std::cerr<<"plan_file_name: "<<plan_file_name<<std::endl;
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

int Model::getLayerNumberFromCalibrationKey(std::string key)
{
	int last_index = key.rfind('_');
	int iter = last_index - 1;
	int number;
	while (isdigit(key.at(iter)) == true)
	{
		iter--;
	}
	std::stringstream ssString(key.substr(iter+1, last_index - (iter + 1)));
	ssString >> number;

	return number;
}


void Model::readFromCalibrationTable(std::string basic_calibration_table, int start_index, int end_index, std::string out_calib_table, int device) {
	std::ifstream input(basic_calibration_table.c_str());
	std::ofstream output(out_calib_table.c_str());
	std::string title;
	std::string key;
	std::string value;
	input >> title;
	//std::cout << title << std::endl;
	output << title << std::endl;
	std::set<int> inputLayerSet = tk::dnn::NetworkRT::getInputLayers(net, start_index, end_index);
	std::vector<CalibrationTable> calibration_vec;

	while(!input.eof())
	{
		input >> key;
		input >> value;
		CalibrationTable calib;
		//std::cout << "key: " << key << ", value: " << value << std::endl;
		if(key == "data:" && start_index > 0)  {
			continue;			
		}
		else if(key == "out:" || key == "data:") {
			//std::cout  << key << " " << value << std::endl;
			calib.key = key;	
			calib.value = value;
			calibration_vec.push_back(calib);
		}
		else {
			int layer_number = getLayerNumberFromCalibrationKey(key);
			if((layer_number >= start_index && layer_number <= end_index) || 
				inputLayerSet.find(layer_number) != inputLayerSet.end()) {
				//std::cout  << key << " " << value << std::endl;
				calib.key = key;	
				calib.value = value;
				calibration_vec.push_back(calib);
			}
			else if(layer_number > end_index) {
				break;
			}
		}

		if(key == "out:") break;
	}
	if(device == DEVICE_DLA) {
		for (std::vector<CalibrationTable>::iterator it = calibration_vec.begin() ; it != calibration_vec.end(); it++) {
			key = (*it).key;
			value = (*it).value;
			if(!(key == "out:" || key == "data:")) {
				int layer_number = getLayerNumberFromCalibrationKey(key);
				if(layer_number == start_index && key.find("Shortcut", 0) == 0 && key.find("poolOut", 0) > 0 ) {
					auto prev_it = it-1;
					(*prev_it).value = value;	
					break;
				}
			}
		}
	}

	for (std::vector<CalibrationTable>::iterator it = calibration_vec.begin() ; it != calibration_vec.end(); it++) {
		key = (*it).key;
		value = (*it).value;
		output << key << " " << value << std::endl;	
	}
}

void Model::createCalibrationTable(std::string plan_file_name, int iter, int start_index, int end_index) {
	int device_num = config_data->instances.at(instance_id).device_num;
	int data_type = config_data->instances.at(instance_id).data_type;
	std::string calib_table = config_data->instances.at(instance_id).calib_table;
	std::string calib_table_name = plan_file_name.substr(0, plan_file_name.rfind('.')) + "-calibration.table";

	if(fileExist(calib_table_name) == false && device_num > 1 && data_type == TYPE_INT8 && 
		fileExist(calib_table) == true) {
		int device = config_data->instances.at(instance_id).devices.at(iter);
		readFromCalibrationTable(calib_table, start_index, end_index, calib_table_name, device);
	}
}

void Model::setDataType() {
	int data_type = config_data->instances.at(instance_id).data_type;

	if(data_type == TYPE_FP32) {
		net->fp16 = false;	
		net->int8 = false;
	}
	else if(data_type == TYPE_FP16) {
		net->fp16 = true;	
		net->int8 = false;
	}
	else if(data_type == TYPE_INT8) {
		net->fp16 = false;	
		net->int8 = true;
	}
}

void Model::initializeModel() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int start_index = 0;

	// parse a network using tkDNN darknetParser

	net = app->createNetwork(&(config_data->instances.at(instance_id)));
	net->print();
	
	letter_box = net->letterBox;
	input_dim.width = net->input_dim.w;
	input_dim.height = net->input_dim.h;
	input_dim.channel = net->input_dim.c;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		std::string plan_file_name;
		int cut_point = config_data->instances.at(instance_id).cut_points[iter1];
		int dla_core = config_data->instances.at(instance_id).dla_cores[iter1];

		getModelFileName(iter1, plan_file_name);
		createCalibrationTable(plan_file_name, iter1, start_index, cut_point);

		setDevice(iter1);
		setMaxBatchSize();
		setDataType();

		int duplication_num = dla_core <= 1/* && (buffer_num <= 4 || net->dla == false)*/  ? 1 : std::max(dla_core, 2); 

		for(int iter2 = 0; iter2 < duplication_num; iter2++) {
			int core = dla_core <= 1 ? dla_core : iter2 % DLA_NUM;

			tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, plan_file_name.c_str(), start_index, cut_point, core);
			assert(netRT->engineRT != nullptr);

			netRTs[iter1].push_back(netRT);
		}
	
		start_index = cut_point + 1;
	}
	net->releaseLayers();
	delete net;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		std::vector<nvinfer1::IExecutionContext *> context_vec;
		int stream_number = config_data->instances.at(instance_id).stream_numbers[iter1];
		for(int iter2 = 0; iter2 < stream_number; iter2++) {
			int size = netRTs[iter1].size();
			int index = size == 1 ? 0 : iter2 % DLA_NUM;

			nvinfer1::IExecutionContext *context = netRTs[iter1][index]->engineRT->createExecutionContext();	
			assert(context);

			context->setProfiler(&profiler);
			context_vec.push_back(context);
		}
		contexts.push_back(context_vec);
	}
}

void Model::finalizeModel() {
	int device_num = config_data->instances.at(instance_id).device_num;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		int stream_number = config_data->instances.at(instance_id).stream_numbers[iter1];
		for(int iter2 = 0; iter2 < stream_number; iter2++) {
			nvinfer1::IExecutionContext *context = contexts[iter1][iter2];
			context->destroy();
		}

		for(unsigned int iter2 = 0; iter2 < netRTs[iter1].size(); iter2++) {
			delete netRTs[iter1][iter2];
		}
	}
}

void Model::setBindingsNum(int curr, int &input_binding_num, int &output_binding_num) {
	input_binding_num = 0;
	output_binding_num = 0;

	for(int iter = 0; iter < netRTs[curr][0]->engineRT->getNbBindings(); iter++) {
		if(netRTs[curr][0]->engineRT->bindingIsInput(iter)) {
			input_binding_num++;
		}	
		else {
			output_binding_num++;	
		}
	}
}

void Model::initializeBindingVariables() {
	int device_num = config_data->instances.at(instance_id).device_num;

	start_bindings.clear();
	binding_size.clear();
	is_net_output.clear();

	start_bindings.push_back(1);
	for(int iter = 1; iter <= device_num; iter++) {
		start_bindings.push_back(-1);
	}

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		int curr_binding_num = netRTs[iter1][0]->engineRT->getNbBindings();
		for(int iter2 = 0; iter2 < curr_binding_num; iter2++) {
			is_net_output.push_back(false);
			binding_size.push_back(0);
		}	
	}
	binding_size.at(0) = input_dim.width * input_dim.height * input_dim.channel;

	total_binding_num = 1;
}

void Model::setBufferIndexing() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int input_binding_num = 0, output_binding_num = 0, curr_binding_num = 0;
	
	initializeBindingVariables();

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		setBindingsNum(iter1, input_binding_num, output_binding_num);
		curr_binding_num = netRTs[iter1][0]->engineRT->getNbBindings();

		start_bindings[iter1] = start_bindings[iter1] - input_binding_num;
		start_bindings[iter1+1] = start_bindings[iter1] + curr_binding_num;


		for(int iter2 = input_binding_num; iter2 < curr_binding_num; iter2++) {
			int index = start_bindings[iter1];

			nvinfer1::Dims dim = netRTs[iter1][0]->engineRT->getBindingDimensions(iter2);	
			binding_size[index + iter2] = dim.d[0] * dim.d[1] * dim.d[2];
			total_binding_num++;
		}
		app->referNetworkRTInfo(iter1, netRTs[iter1][0]);

		if(iter1 == device_num -  1) {
			int index = start_bindings[iter1] + curr_binding_num - output_binding_num;
			for(int iter2 = index; iter2 < index + output_binding_num; iter2++) {
				is_net_output[iter2] = true;
				this->network_output_number++;
			}
		}
	}
}

void Model::allocateStream() {
	int device_num = config_data->instances.at(instance_id).device_num;

	streams.clear();
	events.clear();

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		std::vector<cudaStream_t> stream_vec;
		std::vector<cudaEvent_t> event_vec;
		int stream_number = config_data->instances.at(instance_id).stream_numbers[iter1];
		for(int iter2 = 0; iter2 < stream_number; iter2++) {
			cudaStream_t stream;
			cudaEvent_t event;
			check_error(cudaStreamCreate(&stream));
			check_error(cudaEventCreate(&event));
			stream_vec.push_back(stream);
			event_vec.push_back(event);
		}
		streams.push_back(stream_vec);
		events.push_back(event_vec);

	}
}

void Model::deallocateStream() {
	int device_num = config_data->instances.at(instance_id).device_num;

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		int stream_number = config_data->instances.at(instance_id).stream_numbers[iter1];
		for(int iter2 = 0; iter2 < stream_number; iter2++) {
			cudaStream_t stream = streams[iter1].back();
			cudaStreamDestroy(stream);
			streams[iter1].pop_back();
			cudaEvent_t event = events[iter1].back();
			cudaEventDestroy(event);
		}
		streams[iter1].clear();
		events[iter1].clear();
	}
	streams.clear();
	events.clear();
}

void Model::setStreamBuffer() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int iter1 = 0; iter1 < total_binding_num; iter1++) {
		for(int iter2 = 0; iter2 < buffer_num; iter2++) {
			stream_buffers.push_back(nullptr);	
		}
	}
}

void Model::allocateBuffer() {
	int batch = config_data->instances.at(instance_id).batch;
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int data_type = config_data->instances.at(instance_id).data_type;

	input_buffers.clear();
	output_buffers.clear();

	for(int iter1 = 0; iter1 < buffer_num; iter1++) {
		float *input_buffer = cuda_make_array_host(batch * binding_size[0]);
		cudaHostGetDevicePointer(&(stream_buffers[iter1 * total_binding_num]), input_buffer, 0);
		input_buffers.push_back(input_buffer);

		for(int iter2 = 1; iter2 < total_binding_num; iter2++) {
			if(!is_net_output[iter2]) {
				if(data_type == TYPE_INT8) {
					stream_buffers[iter1 * total_binding_num + iter2] = cuda_make_array_8(NULL, batch * binding_size[iter2]);
				}
				else {
					stream_buffers[iter1 * total_binding_num + iter2] = cuda_make_array_16(NULL, batch * binding_size[iter2]);
				}
			}
			else {
				float *output_buffer = cuda_make_array_host(batch * binding_size[iter2]);
				cudaHostGetDevicePointer(&(stream_buffers[iter1 * total_binding_num + iter2]), output_buffer, 0);
				output_buffers.push_back(output_buffer);
			}
		}	
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

	while(!stream_buffers.empty()) {
		void *buffer = stream_buffers.back();
		if(buffer != nullptr)
			cudaFree(buffer);
		stream_buffers.pop_back();
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

bool Model::checkInputConsumed(int device_id, int stream_id) {
	cudaError_t error = cudaEventQuery(events[device_id][stream_id]);

	if(error == cudaSuccess)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool Model::checkInferenceDone(int device_id, int stream_id) {
	cudaError_t error = cudaStreamQuery(streams[device_id][stream_id]);	

	if(error == cudaSuccess)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void Model::infer( int device_id, int stream_id, int buffer_id) {
	int start_binding = start_bindings[device_id] + total_binding_num * buffer_id;
	int batch = config_data->instances.at(instance_id).batch;
	bool enqueueSuccess = false;
	bool is_dla = (config_data->instances.at(instance_id).devices.at(device_id) == DEVICE_DLA);
	int dla_core = config_data->instances.at(instance_id).dla_cores.at(device_id);
	long start_time;

	if(is_dla) {
		start_time = getTime();	
	}

	// enqueueSuccess = contexts[device_id][stream_id]->enqueue(batch, &(stream_buffers[start_binding]), streams[device_id][stream_id], &(events[device_id][stream_id]));
	enqueueSuccess = contexts[device_id][buffer_id]->execute(batch, &(stream_buffers[start_binding]));
	if(enqueueSuccess == false)
	{
		printf("enqueue error happened: %d, %d\n", device_id, buffer_id);
		// exit_flag = true;
	}

	if(is_dla) {
		long dla_time = getTime() - start_time;	
		dla_profile_vec.at(dla_core).push_back(dla_time);
	}
}

void Model::waitUntilInputConsumed(int device_id, int stream_id) {
	cudaError_t error;

	error = cudaEventSynchronize(events[device_id][stream_id]);
	if(error != cudaSuccess)
	{
		printf("error happened in synchronize: %d, %d: %d\n", device_id, stream_id, error);
		// exit_flag = true;
	}
}

void Model::waitUntilInferenceDone(int device_id, int stream_id) {
	cudaError_t error;

	error = cudaStreamSynchronize(streams[device_id][stream_id]);
	if(error != cudaSuccess)
	{
		printf("error happened in synchronize: %d, %d: %d\n", device_id, stream_id, error);
		// exit_flag = true;
	}
}

void Model::printProfile(char *max_profile_file_name, char *avg_profile_file_name, char *min_profile_file_name) {
	profiler.saveLayerTimes(max_profile_file_name, avg_profile_file_name, min_profile_file_name, dla_profile_vec);
}
