#ifndef IMAGE_DATASET_H_
#define IMAGE_DATASET_H_

#include <iostream>
#include <vector>
#include <cassert>

#include "dataset.h"

typedef struct _ImageData {
  std::string path;
  int width;
  int height;
} ImageData;

class ImageDataset : public Dataset<ImageData> {
	public:
		ImageDataset(std::string imagePathListFile);
		~ImageDataset();
		virtual ImageData *getData(int index) override;
	private:
		std::string imagePathListFile;
		void fillImagePath(char *filename);
};

#endif
