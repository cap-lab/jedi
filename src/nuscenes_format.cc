#include <sys/stat.h>
#include <sys/types.h>

#include "nuscenes_format.h"

void NuscenesFormat::writeResultFile(std::string result_dir_name) {
	std::ofstream result_file;
	struct stat info;
	if (stat(result_dir_name.c_str(), &info) != 0) {
		if (mkdir(result_dir_name.c_str(), 0755) == -1) {
			std::cerr << "Cannot create directory: " << result_dir_name << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	if (info.st_mode & S_IFDIR) {
		std::cout << "Directory is already exists: " << result_dir_name << std::endl;
	} else {
		std::cerr << "The path indicates a file: " << result_dir_name << std::endl;
		exit(EXIT_FAILURE);
	}

	for (const auto& kv : detected_map) {
		std::ofstream result_file;
		std::stringstream result_file_name;

		result_file_name << result_dir_name << "/" << kv.first << ".txt";
		result_file.open(result_file_name.str());

		for (const auto& val : kv.second) {
			result_file << val;
		}
		result_file.close();
	}
}

static std::string extractFileName(const std::string& filePath) {
	size_t lastSlash = filePath.find_last_of("/\\");
	size_t lastDot = filePath.find_last_of('.');
	std::string fileName = filePath.substr(lastSlash + 1, lastDot - lastSlash - 1);
	return fileName;
}


void NuscenesFormat::saveOutput(std::vector<Box>& predResult, std::string &inputFileName) {
	std::string token;
	std::list<std::string> detected;
	int max_obj_num = predResult.size();
	if(max_obj_num > 500) {
		max_obj_num = 500;
	}

	token = extractFileName(inputFileName);
	for (size_t idx = 0; idx < max_obj_num; idx++){
		std::stringstream result;

		result << predResult[idx].x << " " << predResult[idx].y << " " << predResult[idx].z << " "<< \
		predResult[idx].l << " " << predResult[idx].h << " " << predResult[idx].w << " " << predResult[idx].velX \
		<< " " << predResult[idx].velY << " " << predResult[idx].theta << " " << predResult[idx].score << \
		" "<< predResult[idx].cls << std::endl;

		detected.emplace_back(result.str());
	}
	addToDetectedMap(token, detected);
	//std::cout <<token << std::endl;
}


void NuscenesFormat::addToDetectedMap(std::string token, std::list<std::string> detected){
	mu.lock();
	detected_map.insert(std::pair<std::string,std::list<std::string>>(token,detected));
	mu.unlock();
}
