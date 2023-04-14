#ifndef _COCO_FORMAT_H_
#define _COCO_FORMAT_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <mutex>

#include "variable.h"
#include "config.h"
#include "result_format.h"

class COCOFormat : public ResultFormat {
	public:
		COCOFormat() {};
		~COCOFormat() {};
		void writeResultFile(std::string result_file_name) override;
		void detectCOCO(Detection *dets, int nDets, int idx, int w, int h, int iw, int ih, char *path);
		void addToDetectedMap(int image_index, std::list<std::string> detected);

	private:
		int get_coco_image_id(char *filename);

		std::map<int,std::list<std::string>> detected_map;
		std::mutex mu;
};

#endif 
