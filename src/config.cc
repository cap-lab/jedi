#include <iostream>
#include <cstdlib>
#include <libconfig.h++>
#include <cstring>
#include <sstream>

#include "config.h"
#include "variable.h"

#include "inference_application.h"

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

void ConfigData::readNetworkName(Setting &setting, ConfigInstance &config_instance) {
	try{	
		const char *tmp = setting["network_name"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		config_instance.network_name =  data.c_str();

		std::cerr<<"network_name: "<<config_instance.network_name<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'network_name' setting in configuration file." << std::endl;
	}
}

void ConfigData::readModelDir(Setting &setting, ConfigInstance &config_instance) {
	try{	
		const char *tmp = setting["model_dir"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		config_instance.model_dir = data.c_str();

		std::cerr<<"model_dir: "<<config_instance.model_dir<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'model_dir' setting in configuration file." << std::endl;
	}
}

void ConfigData::readBinPath(Setting &setting, ConfigInstance &config_instance) {
	try{	
		const char *tmp = setting["bin_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		config_instance.bin_path = data.c_str();

		std::cerr<<"bin_path: "<<config_instance.bin_path<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'bin_path' setting in configuration file." << std::endl;
	}
}

void ConfigData::readApplicationType(Setting &setting, ConfigInstance &config_instance) {
	try{
		const char *tmp = setting["app_type"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		config_instance.app_type = data.c_str();

		std::cerr<<"app_type: "<< config_instance.app_type << std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'app_type' setting in configuration file. Set YoloApplication as a default." << std::endl;
		config_instance.app_type = "YoloApplication";
	}
}


void ConfigData::readBatch(Setting &setting, ConfigInstance &config_instance){
	try {
		const char *data = setting["batch"];
		config_instance.batch = atoi(data);

		std::cerr<<"batch: "<<config_instance.batch<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'batch' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readBatchThreadNum(Setting &setting, ConfigInstance &config_instance){
	try {
		const char *data = setting["batch_thread_num"];
		config_instance.batch_thread_num = atoi(data);

		std::cerr<<"batch threads: "<<config_instance.batch_thread_num<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'batch_thread_num' setting in configuration file. Set 1 as a default." <<std::endl;
		config_instance.batch_thread_num = 1;
	}
}

void ConfigData::readOffset(Setting &setting, ConfigInstance &config_instance){
	try {
		const char *data = setting["offset"];
		config_instance.offset = atoi(data);

		std::cerr<<"offset: "<<config_instance.offset<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'offset' setting in configuration file. Set 0 as a default." <<std::endl;
		config_instance.offset = 0;
	}
}

void ConfigData::readSampleSize(Setting &setting, ConfigInstance &config_instance){
	try {
		const char *data = setting["sample_size"];
		config_instance.sample_size = atoi(data);

		std::cerr<<"sample_size: "<<config_instance.sample_size<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'sample_size' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readDeviceNum(Setting &setting, ConfigInstance &config_instance){
	try {
		const char *data = setting["device_num"];
		config_instance.device_num = atoi(data);

		std::cerr<<"device_num: "<<config_instance.device_num<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'device_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readPreThreadNum(Setting &setting, ConfigInstance &config_instance){
	try {
		const char *data = setting["pre_thread_num"];
		config_instance.pre_thread_num = atoi(data);

		std::cerr<<"pre_thread_num: "<<config_instance.pre_thread_num<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'pre_thread_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readPostThreadNum(Setting &setting, ConfigInstance &config_instance){
	try {
		const char *data = setting["post_thread_num"];
		config_instance.post_thread_num = atoi(data);

		std::cerr<<"post_thread_num: "<<config_instance.post_thread_num<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'post_thread_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readBufferNum(Setting &setting, ConfigInstance &config_instance){
	try {
		const char *data = setting["buffer_num"];
		config_instance.buffer_num = atoi(data);

		std::cerr<<"buffer_num: "<<config_instance.buffer_num<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'buffer_num' setting in configuration file." <<std::endl;
	}
}

void ConfigData::readCutPoints(Setting &setting, ConfigInstance &config_instance){
	try{
		const char *data = setting["cut_points"];
		std::stringstream ss(data);
		std::string temp;

		while(getline(ss,temp,',')) {
			config_instance.cut_points.push_back(std::stoi(temp));
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'cut_points' setting in configuration file." << std::endl;
	}
}

void ConfigData::readDevices(Setting &setting, ConfigInstance &config_instance){
	try{
		const char *data = setting["devices"];
		std::stringstream ss(data);
		std::string temp;

		while( getline(ss,temp,','))
		{
			if (temp == "GPU"){
				config_instance.devices.push_back(DEVICE_GPU);
			}
			else if(temp == "DLA"){
				config_instance.devices.push_back(DEVICE_DLA);
			}
			else {
				throw "device type is not correct.";
			}
		}
	}
	catch(const SettingNotFoundException &nfex){
		std::cerr << "No 'devices' setting in configuration file." << std::endl;
	}
}

#define DEFAULT_STREAM_NUM (2)

void ConfigData::readStreams(Setting &setting, ConfigInstance &config_instance){
	try{
		const char *data = setting["streams"];
		std::stringstream ss(data);
		std::string temp;

		while( getline(ss,temp,',')) {
			config_instance.stream_numbers.push_back(std::stoi(temp));
		}

		while(config_instance.device_num > (int) config_instance.stream_numbers.size()) 
		{
			std::cerr << "The number of streams is less than the number of devices. Set 2 as a default stream number for the rest of devices." << std::endl;
			config_instance.stream_numbers.push_back(DEFAULT_STREAM_NUM);
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'streams' setting in configuration file. Set 2 as a default stream number." << std::endl;
		for(int iter2 = 0 ; iter2 < config_instance.device_num ; iter2++) {
			config_instance.stream_numbers.push_back(DEFAULT_STREAM_NUM);
		}
	}
}


void ConfigData::readDlaCores(Setting &setting, ConfigInstance &config_instance){
	try{
		const char *data = setting["dla_cores"];
		std::stringstream ss(data);
		std::string temp;

		while( getline(ss,temp,',')) {
			config_instance.dla_cores.push_back(std::stoi(temp));
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'dla_cores' setting in configuration file." << std::endl;
	}
}

void ConfigData::readCalibTable(Setting &setting, ConfigInstance &config_instance) {
	try{	
		if(config_instance.data_type == TYPE_INT8)
		{
			const char *tmp = setting["calib_table"];
			std::stringstream ss(tmp);
			static std::string data;
			ss >> data;
			config_instance.calib_table = data.c_str();
		}
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "Missing 'calib_table' setting in configuration file." << std::endl;
	}

}

void ConfigData::readDataType(Setting &setting, ConfigInstance &config_instance) {
	try{	
		const char *tmp = setting["data_type"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;

		config_instance.data_type = TYPE_FP16;
		if(data == "INT8") {
			config_instance.data_type = TYPE_INT8;
		}
		else if(data == "FP32") {
			config_instance.data_type = TYPE_FP32;
		}

		std::cerr<<"data_type: "<<config_instance.data_type<<std::endl;
	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'data_type' setting in configuration file." << std::endl;
	}
}

ConfigData::ConfigData(std::string config_file_path, std::vector<IInferenceApplication *> &apps) {
	Config cfg;

	readConfigFile(&cfg, config_file_path);
	readInstanceNum(&cfg);

	try{
		Setting &settings = cfg.lookup("configs.instances");

		for(int iter = 0; iter < instance_num; iter++) {
			ConfigInstance config_instance;	
			instances.push_back(config_instance);

			readApplicationType(settings[iter], instances.at(iter));
			readNetworkName(settings[iter], instances.at(iter));
			readModelDir(settings[iter], instances.at(iter));
			readBinPath(settings[iter], instances.at(iter));
			readBatch(settings[iter], instances.at(iter));
			readBatchThreadNum(settings[iter], instances.at(iter));
			readOffset(settings[iter], instances.at(iter));
			readSampleSize(settings[iter], instances.at(iter));
			readDeviceNum(settings[iter], instances.at(iter));
			readPreThreadNum(settings[iter], instances.at(iter));
			readPostThreadNum(settings[iter], instances.at(iter));
			readBufferNum(settings[iter], instances.at(iter));
			readCutPoints(settings[iter], instances.at(iter));
			readDevices(settings[iter], instances.at(iter));
			readDlaCores(settings[iter], instances.at(iter));
			readDataType(settings[iter], instances.at(iter));
			readStreams(settings[iter], instances.at(iter));
			readCalibTable(settings[iter], instances.at(iter));
		}

		for(int iter = 0; iter < instance_num; iter++) {
			IInferenceApplication *app = nullptr;
			app = g_AppRegistry.create(instances.at(iter).app_type);
			app->readCustomOptions(settings[iter]);
			apps.push_back(app);
		}

	}
	catch(const SettingNotFoundException &nfex) {
		std::cerr << "No 'instances' in configuration file." << std::endl;
	}

}

ConfigData::~ConfigData() {
	for(int iter = 0; iter < instance_num; iter++) {
		instances.at(iter).cut_points.clear();
		instances.at(iter).devices.clear();
		instances.at(iter).dla_cores.clear();
	}

	instances.clear();
}
