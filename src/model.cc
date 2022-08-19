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
}

Model::~Model() {
	for(unsigned int iter = 0; iter < stages.size(); iter++) {
		Stage *stage = stages[iter];
		delete stage;
	}
	stages.clear();
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
	output << title << std::endl;
	std::set<int> inputLayerSet = tk::dnn::NetworkRT::getInputLayers(net, start_index, end_index);
	std::vector<CalibrationTable> calibration_vec;

	while(!input.eof())
	{
		input >> key;
		input >> value;
		CalibrationTable calib;
		if(key == "data:" && start_index > 0)  {
			continue;			
		}
		else if(key == "out:" || key == "data:") {
			calib.key = key;	
			calib.value = value;
			calibration_vec.push_back(calib);
		}
		else {
			int layer_number = getLayerNumberFromCalibrationKey(key);
			if((layer_number >= start_index && layer_number <= end_index) || 
				inputLayerSet.find(layer_number) != inputLayerSet.end()) {
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

		Stage *stage = new Stage(config_data, instance_id, iter1, start_index, cut_point);

		int duplication_num = dla_core <= 1 ? 1 : std::max(dla_core, 2); 

		for(int iter2 = 0; iter2 < duplication_num; iter2++) {
			int core = dla_core <= 1 ? dla_core : iter2 % DLA_NUM;

			tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, plan_file_name.c_str(), start_index, cut_point, core);
			assert(netRT->engineRT != nullptr);
			stage->netRTs.push_back(netRT);
		}
	
		stages.push_back(stage);

		start_index = cut_point + 1;
	}

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		Stage *stage = stages[iter1];
		stage->createExecutionContext();
		stage->setInputOutputLayerId(net);

		app->referNetworkRTInfo(iter1, stage->netRTs[0]);
	}

	net->releaseLayers();
	delete net;
}

void Model::finalizeModel() {
	for(unsigned int iter1 = 0; iter1 < stages.size(); iter1++) {
		Stage *stage = stages[iter1];
		stage->finalizeStage();
	}
}

void Model::allocateStream() {
	for(unsigned int iter1 = 0; iter1 < stages.size(); iter1++) {
		Stage *stage = stages[iter1];
		stage->allocateStream();
	}
}

void Model::deallocateStream() {
	for(unsigned int iter1 = 0; iter1 < stages.size(); iter1++) {
		Stage *stage = stages[iter1];
		stage->deallocateStream();
	}
}

void* Model::makeCUDAArray(int size) {
	int data_type = config_data->instances.at(instance_id).data_type;
	void *space;

	if(data_type == TYPE_INT8) {
		space = (void *)cuda_make_array_8(NULL, size);
	}
	else {
		space = (void *)cuda_make_array_16(NULL, size);
	}
	
	return space;
}

void Model::allocateInputStreamBuffer(std::map<std::pair<int, int>, void*>& stream_buffers_map, std::vector<float *>& input_buffer, std::map<std::pair<int, int>, bool*>& signals_map, std::vector<bool*>& input_signal) {
	int batch = config_data->instances.at(instance_id).batch;
	void *space = nullptr;
	int size = input_dim.width * input_dim.height * input_dim.channel;
	bool *signal = new bool(false);

	float *buf = cuda_make_array_host(batch * size);
	cudaHostGetDevicePointer(&(space), buf, 0);
	input_buffer.push_back(buf);
	stream_buffers_map.insert(std::make_pair(std::make_pair(-1, 0), space));

	input_signal.push_back(signal);
	signals_map.insert(std::make_pair(std::make_pair(-1, 0), signal));
}

void* Model::getOutputBufferOfLayer(std::map<std::pair<int, int>, void*>& stream_buffers_map, int tsrc_id) {
	void *space = nullptr;

	for(auto iter = stream_buffers_map.begin(); iter != stream_buffers_map.end(); iter++) {
		auto pair_ids = iter->first;
		int src_id = pair_ids.first;

		if(src_id == tsrc_id) {
			space = iter->second;
			break;
		}
	}

	return space;
}

void Model::allocateStreamBuffer(std::map<std::pair<int, int>, int> size_map, std::map<std::pair<int, int>, void*>& stream_buffers_map, std::vector<float *>& output_buffer, std::map<std::pair<int, int>, bool*>& signals_map, std::vector<bool*>& output_signal) {
	int batch = config_data->instances.at(instance_id).batch;
	void *space = nullptr;
	int size = -1;
	bool *signal = nullptr;

	for(auto iter = size_map.begin(); iter != size_map.end(); iter++) {
		auto pair_ids = iter->first;
		int src_id = pair_ids.first, dst_id = pair_ids.second;

		if(stream_buffers_map.find(std::make_pair(src_id, dst_id)) == stream_buffers_map.end()) {
			size = iter->second;	
			space = nullptr;
			signal = new bool(false);

			if(src_id != -1 && dst_id != -1) {
				space = getOutputBufferOfLayer(stream_buffers_map, src_id);
				if(space == nullptr) {
					space = makeCUDAArray(batch * size);
				}
			}
			else if(dst_id == -1) {
				float *buf = cuda_make_array_host(batch * size);
				cudaHostGetDevicePointer(&(space), buf, 0);
				output_buffer.push_back(buf);
				output_signal.push_back(signal);
			}

			stream_buffers_map.insert(std::make_pair(std::make_pair(src_id, dst_id), space));
			signals_map.insert(std::make_pair(std::make_pair(src_id, dst_id), signal));
		}
	}
}

void Model::allocateBuffer() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int buffer_id = 0; buffer_id < buffer_num; buffer_id++) {
		std::map<std::pair<int, int>, void*> stream_buffers_map;
		std::map<std::pair<int, int>, bool*> signals_map;
		std::vector<float*> input_buffer;
		std::vector<float*> output_buffer;
		std::vector<bool*> input_signal;
		std::vector<bool*> output_signal;

		allocateInputStreamBuffer(stream_buffers_map, input_buffer, signals_map, input_signal);

		for(unsigned int stage_id = 0; stage_id < stages.size(); stage_id++) {
			Stage *stage = stages[stage_id];	

			allocateStreamBuffer(stage->input_size_map, stream_buffers_map, output_buffer, signals_map, output_signal);
			allocateStreamBuffer(stage->output_size_map, stream_buffers_map, output_buffer, signals_map, output_signal);
		}

		net_input_buffers.push_back(input_buffer);
		net_output_buffers.push_back(output_buffer);
		all_stream_buffers.push_back(stream_buffers_map);

		net_input_signals.push_back(input_signal);
		net_output_signals.push_back(output_signal);
		all_signals.push_back(signals_map);

		this->network_output_number = output_buffer.size();
	}
}

void Model::deallocateBuffer() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;

	for(int buffer_id = 0; buffer_id < buffer_num; buffer_id++) {
		auto stream_buffers_map = all_stream_buffers[buffer_id];
		auto signals_map = all_signals[buffer_id];

		for(auto iter = stream_buffers_map.begin(); iter != stream_buffers_map.end(); iter++) {
			auto pair_ids = iter->first;
			int src_id = pair_ids.first, dst_id = pair_ids.second;

			if(src_id == -1 || dst_id == -1) {
				void *buffer = iter->second;
				if(buffer != nullptr)
					cudaFreeHost(buffer);
			}
			else {
				void *buffer = iter->second;
				if(buffer != nullptr)
					cudaFree(buffer);
			}
		}

		for(auto iter = signals_map.begin(); iter != signals_map.end(); iter++) {
			delete iter->second;
		}
	}
}

void Model::setBufferForStage() {
	int buffer_num = config_data->instances.at(instance_id).buffer_num;
	int device_num = config_data->instances.at(instance_id).device_num;

	for(int iter1 = 0; iter1 < buffer_num; iter1++) {
		for(int iter2 = 0; iter2 < device_num; iter2++) {
			Stage *stage = stages[iter2];
			stage->setBuffers(iter1, all_stream_buffers[iter1]);
			stage->setSignals(iter1, all_signals[iter1]);
		}
	}
}

bool Model::isPreprocessingRunnable(int buffer_id) {
	std::vector<bool*> input_signal = net_input_signals[buffer_id];

	for(unsigned int iter = 0; iter < input_signal.size(); iter++) {
		if(*(input_signal[iter]) == true)	
			return false;
	}

	return true;
}

bool Model::isPostprocessingRunnable(int buffer_id) {
	std::vector<bool*> output_signal = net_output_signals[buffer_id];

	for(unsigned int iter = 0; iter < output_signal.size(); iter++) {
		if(*(output_signal[iter]) == false)	
			return false;
	}

	return true;
}

void Model::updateInputSignals(int buffer_id, bool value) {
	std::vector<bool*> input_signal = net_input_signals[buffer_id];

	for(unsigned int iter = 0; iter < input_signal.size(); iter++) {
		*(input_signal[iter]) = value;	
	}
}

void Model::updateOutputSignals(int buffer_id, bool value) {
	std::vector<bool*> output_signal = net_output_signals[buffer_id];

	for(unsigned int iter = 0; iter < output_signal.size(); iter++) {
		*(output_signal[iter]) = value;	
	}
}

void Model::initializeBuffers() {
	allocateStream();
	allocateBuffer();
	setBufferForStage();
}

void Model::finalizeBuffers() {
	deallocateBuffer();
	deallocateStream();
}

bool Model::checkInputConsumed(int device_id, int stream_id) {
	Stage *stage = stages[device_id];
	cudaError_t error = cudaEventQuery(stage->events[stream_id]);

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
	Stage *stage = stages[device_id];
	cudaError_t error = cudaStreamQuery(stage->streams[stream_id]);	

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
	Stage *stage = stages[device_id];
	int batch = config_data->instances.at(instance_id).batch;
	bool enqueueSuccess = false;

	enqueueSuccess = stage->contexts[stream_id]->enqueue(batch, &(stage->stage_buffers[buffer_id][0]), stage->streams[stream_id], &(stage->events[stream_id]));
	// enqueueSuccess = stage->contexts[stream_id]->execute(batch, &(stage->stage_buffers[buffer_id][0]));
	if(enqueueSuccess == false)
	{
		printf("enqueue error happened: %d, %d\n", device_id, buffer_id);
		exit_flag = true;
	}
}

void Model::waitUntilInputConsumed(int device_id, int stream_id) {
	Stage *stage = stages[device_id];
	cudaError_t error;

	error = cudaEventSynchronize(stage->events[stream_id]);
	if(error != cudaSuccess)
	{
		printf("error happened in synchronize: %d, %d: %d\n", device_id, stream_id, error);
		exit_flag = true;
	}
}

void Model::waitUntilInferenceDone(int device_id, int stream_id) {
	Stage *stage = stages[device_id];
	cudaError_t error;

	error = cudaStreamSynchronize(stage->streams[stream_id]);
	if(error != cudaSuccess)
	{
		printf("error happened in synchronize: %d, %d: %d\n", device_id, stream_id, error);
		exit_flag = true;
	}
}
