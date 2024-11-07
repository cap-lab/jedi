#include <float.h>

#include "imagenet_format.h"

#define CLASS_NUM 1000

ImagenetFormat::ImagenetFormat() {
	class_num = CLASS_NUM;
}

void ImagenetFormat::writeResultFile(std::string result_file_name) {
	int total_num = right_num + wrong_num;

	fprintf(stderr, "Prediction Accuracy : %.2f(%d of %d)\n", ((float)right_num) / (right_num + wrong_num) * 100, right_num, wrong_num + right_num);
}

void ImagenetFormat::recordIsCorrect(bool is_correct) {
	mu.lock();
	if(is_correct) {
		right_num += 1;	
	}
	else {
		wrong_num += 1;	
	}
	mu.unlock();
}
