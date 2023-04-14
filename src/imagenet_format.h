#ifndef _IMAGENET_H_
#define _IMAGENET_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>

#include "variable.h"
#include "config.h"
#include "result_format.h"

class ImagenetFormat : public ResultFormat {
	public:
		ImagenetFormat();
		~ImagenetFormat() {};
		void writeResultFile(std::string result_file_name) override;
		void recordIsCorrect(bool is_correct);
	
		int class_num = 1000;

	private:
		int wrong_num = 0;
		int right_num = 0;
		std::mutex mu;
};

#endif
