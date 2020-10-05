#include <iostream>
#include <vector>

#include "image.h"
#include "image_opencv.h"
#include "variable.h"

typedef void* mat_cv;

char *fgetl(FILE *fp) {
	if(feof(fp)) return 0;
	size_t size = 512;
	char *line = (char *)malloc(size*sizeof(char));
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
				printf("%ld\n", size);
				fprintf(stderr, "Malloc error\n");
				exit(-1);
			}
		}
		size_t readsize = size-curr;
		if(readsize > INT_MAX) readsize = INT_MAX-1;
		fgets(&line[curr], readsize, fp);
		curr = strlen(line);
	}
	if(line[curr-1] == '\n') line[curr-1] = '\0';

	return line;
}

void getPaths(char *filename, std::vector<std::string> &paths) {
	char *path;
	FILE *file = fopen(filename, "r");
	if(!file) {
		fprintf(stderr, "Couldn't open file: %s\n", filename);
		exit(0);
	}

	while((path=fgetl(file))){
		paths.emplace_back(std::string(path));
		// std::cerr<<__func__<<":"<<__LINE__<<" path: "<<std::string(path)<<" "<<std::endl;
	}
	fclose(file);
}
