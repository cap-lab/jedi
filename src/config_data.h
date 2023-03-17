#ifndef CONFIG_DATA_H_
#define CONFIG_DATA_H_


#include <iostream>
#include <vector>


typedef struct _ConfigInstance {
	std::string network_name;
	std::string app_type;
	std::string model_type;
	std::string model_dir;
	std::string bin_path;
	std::string calib_table;
	int batch;
	int batch_thread_num;
	int offset;
	int sample_size;

	int device_num;
	int pre_thread_num;
	int post_thread_num;
	int buffer_num;
	std::vector<int> cut_points;
	std::vector<int> devices;
	std::vector<int> dla_cores;
	std::vector<int> stream_numbers;
	std::vector<int> data_types;
} ConfigInstance;


#endif
