#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <string>
#include <vector>

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} Image;

void getPaths(char *filename, std::vector<std::string> &paths);

#endif 
