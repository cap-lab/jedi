#ifndef _PILLAR_H_
#define _PILLAR_H_

#define X_STEP 0.2f
#define Y_STEP 0.2f
#define X_MIN -51.2f
#define X_MAX 51.2f
#define Y_MIN -51.2f
#define Y_MAX 51.2f
#define Z_MIN -5.0f
#define Z_MAX 3.0f
#define PI 3.141592653f

// paramerters for preprocess
#define BEV_W 512
#define BEV_H 512
#define MAX_PILLARS 30000
#define MAX_POINT_IN_PILLARS 20
#define FEATURE_NUM 10
#define THREAD_NUM 1

bool readBinFile(std::string& filename, float*& bufPtr, int& pointNum, int &bufSize);
void makePillars(float* points, float* feature, int* indices, int pointNum, int threadIdx, int pillarsPerThread);

#endif