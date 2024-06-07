#ifndef _NUSCENES_FORMAT_H_
#define _NUSCENES_FORMAT_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <mutex>

#include "variable.h"
#include "config.h"
#include "result_format.h"

struct Box{
    float x;
    float y;
    float z;
    float l;
    float h;
    float w;
    float velX;
    float velY;
    float theta;

    float score;
    int cls;
    bool isDrop; // for nms
};


class NuscenesFormat : public ResultFormat {
	public:
		NuscenesFormat() {};
		~NuscenesFormat() {};
		void writeResultFile(std::string result_dir_name) override;
		void addToDetectedMap(std::string token, std::list<std::string> detected);
		void saveOutput(std::vector<Box>& predResult, std::string &inputFileName);

	private:
        std::map<std::string,std::list<std::string>> detected_map;
		std::mutex mu;
};

#endif 
