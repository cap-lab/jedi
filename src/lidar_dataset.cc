/*
 * lidar_dataset.cc
 *
 *  Created on: 2024. 5. 30.
 *      Author: jej
 */

#include <string.h>
#include <climits>

#include "lidar_dataset.h"


static char *fgetl(FILE *fp) {
	if(feof(fp)) return 0;
	size_t size = 512;
	char *line = (char *)calloc(sizeof(char), size);
	if(!fgets(line, size, fp)){
		free(line);
		return 0;
	}

	size_t curr = strlen(line);

	while((line[curr-1] != '\n') && !feof(fp)){
		if(curr == size-1){
			size *= 2;
			line = (char *)realloc(line, size*sizeof(char));
			if(!line) {
				std::cerr<<size<<std::endl;
				std::cerr<<"Malloc error"<<std::endl;
				exit(-1);
			}
		}
		size_t readsize = size-curr;
		if(readsize > INT_MAX) readsize = INT_MAX-1;
		if(!fgets(&line[curr], readsize, fp)) {
			continue;
		}
		curr = strlen(line);
	}
	if(line[curr-1] == '\n') line[curr-1] = '\0';

	return line;
}


void LidarDataset::fillLidarPath(char *filename) {
	char *path;
	FILE *file = fopen(filename, "r");
	if(!file) {
		std::cerr<<"Couldn't open file: "<<filename<<std::endl;
		exit(0);
	}

	while((path=fgetl(file))){
		LidarData LidarData;
		LidarData.path = std::string(path);
		data.emplace_back(LidarData);
		free(path);
		path = nullptr;
	}
	fclose(file);
}

LidarDataset::LidarDataset(std::string lidarBinListFile)
{
	this->lidarBinListFile = lidarBinListFile;

	fillLidarPath((char *)lidarBinListFile.c_str());
}

LidarDataset::~LidarDataset()
{
	while(!data.empty()) {
		data.pop_back();
	}
}

LidarData *LidarDataset::getData(int index)
{
	return &(data[index]);
}



