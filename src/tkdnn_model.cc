#include <iostream>
#include <vector>
#include <cassert>
#include <cctype>

#include <NvInfer.h>
#include <tkDNN/tkdnn.h>
#include <tkDNN/DarknetParser.h>

#include "cuda.h"
#include "variable.h"

#include "util.h"

#include "tkdnn_network.h"

#include "tkdnn_model.h"

typedef struct _CalibrationTable {
	std::string key;
	std::string value;
} CalibrationTable;


REGISTER_JEDI_NETWORK_MODEL(TkdnnModel);

void TkdnnModel::getModelFileName(int curr, std::string &plan_file_name, int input_width, int input_height) {
	std::string model_dir = config_data->instances.at(instance_id).model_dir;
	std::string cut_points_name;
	std::string device_name;
	std::string data_type_name;
	std::string image_size_name;
	int device = config_data->instances.at(instance_id).devices.at(curr);
	int batch = config_data->instances.at(instance_id).batch;
	int data_type = config_data->instances.at(instance_id).data_types.at(curr);
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

	image_size_name = std::to_string(input_width) + "x" + std::to_string(input_height);

	plan_file_name = model_dir + "/model" + image_size_name + "_" + cut_points_name + "_" + device_name + "_" + data_type_name + "_" + std::to_string(batch) + ".rt";
	std::cerr<<"plan_file_name: "<<plan_file_name<<std::endl;
}

void TkdnnModel::setDevice(int curr) {
	int device = config_data->instances.at(instance_id).devices.at(curr);

	if(device == DEVICE_DLA) {
		net->dla = true;	
	}
	else {
		net->dla = false;	
	}
}

void TkdnnModel::setMaxBatchSize() {
	int batch = config_data->instances.at(instance_id).batch;

	net->maxBatchSize = batch;
}

int TkdnnModel::getLayerNumberFromCalibrationKey(std::string key)
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


void TkdnnModel::readFromCalibrationTable(std::string basic_calibration_table, int start_index, int end_index, std::string out_calib_table, int device) {
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

void TkdnnModel::createCalibrationTable(std::string plan_file_name, int iter, int start_index, int end_index) {
	int device_num = config_data->instances.at(instance_id).device_num;
	int data_type = config_data->instances.at(instance_id).data_types.at(iter);
	std::string calib_table = config_data->instances.at(instance_id).calib_table;
	std::string calib_table_name = plan_file_name.substr(0, plan_file_name.rfind('.')) + "-calibration.table";

	if(fileExist(calib_table_name) == false && device_num > 1 && data_type == TYPE_INT8 && 
		fileExist(calib_table) == true) {
		int device = config_data->instances.at(instance_id).devices.at(iter);
		readFromCalibrationTable(calib_table, start_index, end_index, calib_table_name, device);
	}
}

void TkdnnModel::setDataType(int device_id) {
	int data_type = config_data->instances.at(instance_id).data_types.at(device_id);

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

/*void TkdnnModel::setInputOutputLayerId(tk::dnn::Network *net, int start_index, int end_index) {
	input_size_maps.push_back(tk::dnn::NetworkRT::getInputPair(net, start_index, end_index));
	output_size_maps.push_back(tk::dnn::NetworkRT::getOutputPair(net, start_index, end_index));
}*/


void TkdnnModel::initializeModel() {
	int device_num = config_data->instances.at(instance_id).device_num;
	int start_index = 0;
	int batch = config_data->instances.at(instance_id).batch;

	// parse a network using tkDNN darknetParser
	tkdnn_network = dynamic_cast<TkdnnNetwork *>(app->createNetwork(&(config_data->instances.at(instance_id))));
	net = tkdnn_network->net;
	net->print();
	
	//input_dim.width = net->input_dim.w;
	//input_dim.height = net->input_dim.h;
	//input_dim.channel = net->input_dim.c;
	//
	total_input_size = net->input_dim.w * net->input_dim.h * net->input_dim.c * batch;
	setMaxBatchSize();

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		std::string plan_file_name;
		int cut_point = config_data->instances.at(instance_id).cut_points[iter1];
		int dla_core = config_data->instances.at(instance_id).dla_cores[iter1];

		setDataType(iter1);
		getModelFileName(iter1, plan_file_name, net->input_dim.w, net->input_dim.h);
		createCalibrationTable(plan_file_name, iter1, start_index, cut_point);

		setDevice(iter1);
		Stage *stage = new Stage(config_data, instance_id, iter1, start_index, cut_point);

		int duplication_num = dla_core <= 1 ? 1 : std::max(dla_core, 2); 

		for(int iter2 = 0; iter2 < duplication_num; iter2++) {
			int core = dla_core <= 1 ? dla_core : iter2 % DLA_NUM;

			tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, plan_file_name.c_str(), start_index, cut_point, core);
			assert(netRT->engineRT != nullptr);
			stage->engines.push_back(netRT->engineRT);
			netRTs.push_back(netRT);
		}
		//setInputOutputLayerId(net, start_index, cut_point);
		stages.push_back(stage);

		start_index = cut_point + 1;
	}

	for(int iter1 = 0; iter1 < device_num; iter1++) {
		Stage *stage = stages[iter1];
		stage->createExecutionContext();

		// stage->getBindingsDataType();
	}

	net->releaseLayers();
	delete net;
	delete tkdnn_network;
}

void TkdnnModel::finalizeModel() {
	for(unsigned int iter1 = 0; iter1 < stages.size(); iter1++) {
		Stage *stage = stages[iter1];
		stage->finalizeStage();
	}

	for(unsigned int iter1 = 0; iter1 < netRTs.size(); iter1++) {
		delete netRTs[iter1];
	}
}


