#include <iostream>
#include <cstdlib>
#include <libconfig.h++>
#include <cstring>
#include <sstream>

#include "config.h"
#include "variable.h"

using namespace libconfig;

void ConfigData::readConfigFile(Config *cfg, std::string config_file_path) {
	try {
		cfg->readFile(config_file_path.c_str());
	}
	catch(const FileIOException &fioex) {
		std::cerr << "I/O error while reading file." << std::endl;
		exit(0);
	}
	catch(const ParseException &pex) {
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()<< " - " << pex.getError() << std::endl;
		exit(-1);
	}
}

void ConfigData::readInstanceNum(Config *cfg) {
	instance_num = 0;

	try {
		const char *data = cfg->lookup("configs.instance_num");	
		instance_num = atoi(data);
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'instance_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readNetworkName(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["network_name"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			instances.at(iter).network_name = data.c_str();

			std::cerr<<"network_name: "<<instances.at(iter).network_name<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'network_name' setting in configuration file." << std::endl;
	}
}

void ConfigData::readModelDir(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["model_dir"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			instances.at(iter).model_dir = data.c_str();

			std::cerr<<"model_dir: "<<instances.at(iter).model_dir<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'model_dir' setting in configuration file." << std::endl;
	}
}

void ConfigData::readBinPath(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["bin_path"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			instances.at(iter).bin_path = data.c_str();

			std::cerr<<"bin_path: "<<instances.at(iter).bin_path<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'bin_path' setting in configuration file." << std::endl;
	}
}

void ConfigData::readCfgPath(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["cfg_path"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			instances.at(iter).cfg_path = data.c_str();

			std::cerr<<"cfg_path: "<<instances.at(iter).cfg_path<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'cfg_path' setting in configuration file." << std::endl;
	}
}

void ConfigData::readImagePath(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["image_path"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			instances.at(iter).image_path = data.c_str();

			std::cerr<<"image_path: "<<instances.at(iter).image_path<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'image_path' setting in configuration file." << std::endl;
	}
}

void ConfigData::readCalibImagePath(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["calib_image_path"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			instances.at(iter).calib_image_path = data.c_str();

			std::cerr<<"calib_image_path: "<<instances.at(iter).calib_image_path<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_image_path' setting in configuration file." << std::endl;
	}
}

void ConfigData::readCalibImageLabelPath(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["calib_image_label_path"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			instances.at(iter).calib_image_label_path = data.c_str();
			std::cerr<<"calib_image_label_path: "<<instances.at(iter).calib_image_label_path<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_image_label_path' setting in configuration file." << std::endl;
	}
}

void ConfigData::readCalibImagesNum(Config * cfg){
	try {
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["calib_images_num"];
			instances.at(iter).calib_images_num = atoi(data);

			std::cerr<<"calib_images_num: "<<instances.at(iter).calib_images_num<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_images_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readNamePath(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["name_path"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			instances.at(iter).name_path = data.c_str();

			std::cerr<<"name_path: "<<instances.at(iter).name_path<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'name_path' setting in configuration file." << std::endl;
	}
}

void ConfigData::readBatch(Config * cfg){
	try {
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["batch"];
			instances.at(iter).batch = atoi(data);

			std::cerr<<"batch: "<<instances.at(iter).batch<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'batch' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readOffset(Config * cfg){
	try {
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["offset"];
			instances.at(iter).offset = atoi(data);

			std::cerr<<"offset: "<<instances.at(iter).offset<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'offset' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readSampleSize(Config * cfg){
	try {
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["sample_size"];
			instances.at(iter).sample_size = atoi(data);

			std::cerr<<"sample_size: "<<instances.at(iter).sample_size<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'sample_size' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readDeviceNum(Config * cfg){
	try {
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["device_num"];
			instances.at(iter).device_num = atoi(data);

			std::cerr<<"device_num: "<<instances.at(iter).device_num<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'device_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readPreThreadNum(Config * cfg){
	try {
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["pre_thread_num"];
			instances.at(iter).pre_thread_num = atoi(data);

			std::cerr<<"pre_thread_num: "<<instances.at(iter).pre_thread_num<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'pre_thread_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readPostThreadNum(Config * cfg){
	try {
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["post_thread_num"];
			instances.at(iter).post_thread_num = atoi(data);

			std::cerr<<"post_thread_num: "<<instances.at(iter).post_thread_num<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'post_thread_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readBufferNum(Config * cfg){
	try {
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["buffer_num"];
			instances.at(iter).buffer_num = atoi(data);

			std::cerr<<"buffer_num: "<<instances.at(iter).buffer_num<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'buffer_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readCutPoints(Config *cfg){
	try{
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["cut_points"];
			std::stringstream ss(data);
			std::string temp;

			while(getline(ss,temp,',')) {
				instances.at(iter).cut_points.push_back(std::stoi(temp));
			}
		}

	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'cut_points' setting in configuration file." << std::endl;
	}
}

void ConfigData::readDevices(Config *cfg){
	try{
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["devices"];
			std::stringstream ss(data);
			std::string temp;

			while( getline(ss,temp,','))
			{
				if (temp == "GPU"){
					instances.at(iter).devices.push_back(DEVICE_GPU);
				}
				else if(temp == "DLA"){
					instances.at(iter).devices.push_back(DEVICE_DLA);
				}
				else {
					throw "device type is not correct.";
				}
			}
		}
	}
	catch(const SettingNotFoundException &nfex){
		std::cerr << "No 'devices' setting in configuration file." << std::endl;
	}
}

void ConfigData::readStreams(Config *cfg){
	try{
		Setting &settings = cfg->lookup("configs.instances");
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["streams"];
			std::stringstream ss(data);
			std::string temp;

			while( getline(ss,temp,',')) {
				instances.at(iter).stream_numbers.push_back(std::stoi(temp));
			}
		}

	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'streams' setting in configuration file." << std::endl;
		for(int iter = 0; iter < instance_num; iter++) {
			for(int iter2 = 0 ; iter2 < instances.at(iter).device_num ; iter2++) {
				instances.at(iter).stream_numbers.push_back(instances.at(iter).buffer_num);
			}
		}
	}
}


void ConfigData::readDlaCores(Config *cfg){
	try{
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *data = settings[iter]["dla_cores"];
			std::stringstream ss(data);
			std::string temp;

			while( getline(ss,temp,',')) {
				instances.at(iter).dla_cores.push_back(std::stoi(temp));
			}
		}

	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'dla_cores' setting in configuration file." << std::endl;
	}
}

void ConfigData::readDataType(Config *cfg) {
	try{	
		Setting &settings = cfg->lookup("configs.instances");	
		for(int iter = 0; iter < instance_num; iter++) {
			const char *tmp = settings[iter]["data_type"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;

			instances.at(iter).data_type = TYPE_FP16;
			if(data == "INT8") {
				instances.at(iter).data_type = TYPE_INT8;
			}
			else if(data == "FP32") {
				instances.at(iter).data_type = TYPE_FP32;
			}

			std::cerr<<"data_type: "<<instances.at(iter).data_type<<std::endl;
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'data_type' setting in configuration file." << std::endl;
	}
}

ConfigData::ConfigData(std::string config_file_path) {
	Config cfg;

	readConfigFile(&cfg, config_file_path);
	readInstanceNum(&cfg);

	for(int iter = 0; iter < instance_num; iter++) {
		ConfigInstance config_instance;	
		instances.push_back(config_instance);
	}

	readNetworkName(&cfg);
	readModelDir(&cfg);
	readBinPath(&cfg);
	readCfgPath(&cfg);
	readImagePath(&cfg);
	readCalibImagePath(&cfg);
	readCalibImageLabelPath(&cfg);
	readCalibImagesNum(&cfg);
	readNamePath(&cfg);
	readBatch(&cfg);
	readOffset(&cfg);
	readSampleSize(&cfg);
	readDeviceNum(&cfg);
	readPreThreadNum(&cfg);
	readPostThreadNum(&cfg);
	readBufferNum(&cfg);
	readCutPoints(&cfg);
	readDevices(&cfg);
	readDlaCores(&cfg);
	readDataType(&cfg);
	readStreams(&cfg);
}

ConfigData::~ConfigData() {
	for(int iter = 0; iter < instance_num; iter++) {
		instances.at(iter).cut_points.clear();
		instances.at(iter).devices.clear();
		instances.at(iter).dla_cores.clear();
	}

	instances.clear();
}
