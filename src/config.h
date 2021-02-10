#ifndef CONFIG_H_
#define CONFIG_H_         

#include <iostream>
#include <vector>

#include <libconfig.h++>

typedef struct _ConfigInstance {
	std::string network_name;
	std::string model_dir;
	std::string bin_path;
	std::string cfg_path;
	std::string image_path;
	std::string calib_image_path;
	std::string calib_image_label_path;
	int calib_images_num;
	std::string name_path;
	int batch;
	int offset;
	int sample_size;

	int device_num;
	int pre_thread_num;
	int post_thread_num;
	int buffer_num;
	std::vector<int> cut_points;
	std::vector<int> devices;
	std::vector<int> dla_cores;
	int data_type;
} ConfigInstance;

class ConfigData {
	public:
		int instance_num;	
		std::vector<ConfigInstance> instances;

		ConfigData(std::string config_file_path);
		~ConfigData();
	
	private:
		void readConfigFile(libconfig::Config *cfg, std::string config_file_path);
		void readInstanceNum(libconfig::Config *cfg);
		void readNetworkName(libconfig::Config *cfg);
		void readModelDir(libconfig::Config *cfg);
		void readBinPath(libconfig::Config *cfg);
		void readCfgPath(libconfig::Config *cfg);
		void readImagePath(libconfig::Config *cfg);
		void readCalibImagePath(libconfig::Config *cfg);
		void readCalibImageLabelPath(libconfig::Config *cfg);
		void readCalibImagesNum(libconfig::Config * cfg);
		void readNamePath(libconfig::Config *cfg);
		void readBatch(libconfig::Config * cfg);
		void readOffset(libconfig::Config * cfg);
		void readSampleSize(libconfig::Config * cfg);
		void readDeviceNum(libconfig::Config * cfg);
		void readPreThreadNum(libconfig::Config * cfg);
		void readPostThreadNum(libconfig::Config * cfg);
		void readBufferNum(libconfig::Config * cfg);
		void readCutPoints(libconfig::Config *cfg);
		void readDevices(libconfig::Config *cfg);
		void readDlaCores(libconfig::Config *cfg);
		void readDataType(libconfig::Config *cfg);
};

#endif
