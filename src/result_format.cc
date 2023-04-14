#include "result_format.h"

void ResultFormat::createImageDataset(std::string image_path) {
	image_dataset = new ImageDataset(image_path);
}
