#ifndef CONFIG_H_
#define CONFIG_H_

#include <iostream>
#include <vector>

#include "config_data.h"
#include "inference_application.h"

class ConfigData {
	public:
		int instance_num;	
		std::vector<ConfigInstance> instances;

		ConfigData(std::string config_file_path, std::vector<IInferenceApplication *> &apps);
		~ConfigData();
	
	private:
		void readConfigFile(libconfig::Config *cfg, std::string config_file_path);
		void readInstanceNum(libconfig::Config *cfg);
		void readNetworkName(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readModelDir(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readBinPath(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readBatch(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readBatchThreadNum(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readOffset(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readSampleSize(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readDeviceNum(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readPreThreadNum(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readPostThreadNum(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readBufferNum(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readCutPoints(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readDevices(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readDlaCores(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readDataType(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readStreams(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readCalibTable(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readApplicationType(libconfig::Setting &setting, ConfigInstance &config_instance);
		void readNetworkModelType(libconfig::Setting &setting, ConfigInstance &config_instance);
};

#endif
