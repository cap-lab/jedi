#include <float.h>

#include "imagenet_format.h"

#define CLASS_NUM 1000

ImagenetFormat::ImagenetFormat() {
	class_num = CLASS_NUM;
}

void ImagenetFormat::writeResultFile(std::string result_file_name) {
    int total_num = right_num + wrong_num;
    float accuracy = ((float)right_num) / total_num * 100;

    fprintf(stderr, "Prediction Accuracy : %.2f(%d of %d)\n", accuracy, right_num, total_num);

    FILE *file = fopen(result_file_name.c_str(), "w");
    if (file != NULL) {
        fprintf(file, "%.2f", accuracy);
        fclose(file);
    } else {
        fprintf(stderr, "Error opening file %s\n", result_file_name.c_str());
    }
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
