#include <iostream>
#include <fstream>
#include <algorithm>
#include <string.h>
#include <zlib.h>

#include "pillar.h"

/*struct PointsArray {
    float x;
    float y;
    float z;
    float instance;
    float time_lag;
};*/


bool readGzBinFile(std::string& filename, float*& bufPtr, int& pointNum, int &bufSize)
{
    int fileSize;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.close();

    // gz 파일 열기
    gzFile gzfile = gzopen(filename.c_str(), "rb");
    if (!gzfile) {
        std::cerr << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }

    // 파일 크기 추정을 위한 임시 버퍼
    const int tempBufSize = 1 << 18; // 256KB
    char buffer[tempBufSize];
    int bytesRead = 0;
    int totalBytes = 0;
    char * bufCharPtr;

    fileSize *= 2;

    if(bufSize < fileSize) {
        if(bufPtr != nullptr) {
            free(bufPtr);
        }
        bufPtr = (float *) malloc(fileSize);
        if(bufPtr == nullptr){
            std::cerr << "[Error] Malloc Memory Failed! Size: " << fileSize << std::endl;
            return false;
        }
        bufSize = fileSize;
    }

    bufCharPtr = (char *) bufPtr;

    // 파일 내용을 읽어서 임시 버퍼에 저장
    while ((bytesRead = gzread(gzfile, buffer, tempBufSize)) > 0) {
        if(bufSize < totalBytes + bytesRead) {
            bufSize += fileSize;
            bufPtr = (float *) realloc(bufPtr, bufSize);
            if(bufPtr == nullptr){
                std::cerr << "[Error] Realloc Memory Failed! Size: " << bufSize << std::endl;
                return false;
            }
            bufCharPtr = (char *) bufPtr;
        }
        memcpy(bufCharPtr + totalBytes, buffer, bytesRead);
        totalBytes += bytesRead;
    }

    gzclose(gzfile);

    constexpr int featureNum = 5;
    pointNum = totalBytes /sizeof(float) / featureNum;
    if( totalBytes /sizeof(float) % featureNum != 0){
        std::cerr << "[Error] File Size Error! " << totalBytes << std::endl;
    }
    //std::cout << "[INFO] pointNum : " << pointNum << ", filename: " << filename << std::endl;
    return true;
}

bool readBinFile(std::string& filename, float*& bufPtr, int& pointNum, int &bufSize)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        std::cerr << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if(bufSize < fileSize) {
        if(bufPtr != nullptr) {
            free(bufPtr);
        }
        bufPtr = (float *) malloc(fileSize);
        if(bufPtr == nullptr){
            std::cerr << "[Error] Malloc Memory Failed! Size: " << fileSize << std::endl;
            return false;
        }
        bufSize = fileSize;
    }

    // read the data:
    file.read((char*) bufPtr, fileSize);
    file.close();
    
    constexpr int featureNum = 5;
    pointNum = fileSize /sizeof(float) / featureNum;
    if( fileSize /sizeof(float) % featureNum != 0){
        std::cerr << "[Error] File Size Error! " << fileSize << std::endl;
    }
    //std::cout << "[INFO] pointNum : " << pointNum << ", filename: " << filename << std::endl;
    return true;
}

/*static bool pointsCompare(struct PointsArray a, struct PointsArray b) {
    float distance_a = a.x * a.x + a.y * a.y + a.z * a.z;
    float distance_b = b.x * b.x + b.y * b.y + b.z * b.z;

    return distance_a < distance_b;
}*/

void makePillars(float* points, float* feature, int* indices, int pointNum, int threadIdx, int pillarsPerThread){
    // 0 ~ MAX_POINT_IN_PILLARS
    unsigned short pointCount[MAX_PILLARS] = {0};
    //struct PointsArray *pointsArray;

    // 0 ~ MAX_PILLARS
    int pillarsIndices[BEV_W*BEV_H] = {0};
    int pillarCount = threadIdx*pillarsPerThread;

    memset(pillarsIndices, -1, BEV_W*BEV_H*sizeof(int));

    //pointsArray = (struct PointsArray *) points;
    //std::sort(pointsArray, pointsArray + pointNum, pointsCompare);

    for(int idx = 0; idx < pointNum; idx++){
        
        auto x = points[idx*5];
        auto y = points[idx*5+1];
        auto z = points[idx*5+2];
        if(x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX || 
           z < Z_MIN || z > Z_MAX)
           continue;

        int xIdx = int((x-X_MIN)/X_STEP);
        int yIdx = int((y-Y_MIN)/Y_STEP);
        
        if(xIdx % THREAD_NUM != threadIdx)
            continue;

        int pillarIdx = yIdx*BEV_W+xIdx;

        if(pillarIdx >= BEV_W*BEV_H || pillarIdx < 0) {
            std::cout << "xIdx: " << pillarIdx << ", yIdx: " << yIdx << std::endl;
            std::cout << "pillarsIndices[" << pillarIdx << "]: " << std::endl;
        }

        auto pillarCountIdx = pillarsIndices[pillarIdx];

        if (pillarCountIdx >= MAX_PILLARS || (pillarCountIdx == -1 && pillarCount*2 + 1 >= MAX_PILLARS*2)) {
            continue;
        }

        // new pillar index
        if(pillarCountIdx == -1){
            pillarCountIdx = pillarCount;
            /*if(pointCount[pillarCountIdx] > MAX_POINT_IN_PILLARS - 1) {
                continue;
            }*/
            pillarsIndices[pillarIdx] = pillarCount;
            indices[pillarCount*2 + 1] = pillarIdx;
            ++pillarCount;
        }


        // if(pillarCountIdx < 0 || pillarCountIdx >= MAX_PILLARS)  {
        //     std::cout << "pointCount[" << pillarCountIdx << "]: " << pointCount[pillarCountIdx] << std::endl;
        // }

        auto pointNumInPillar = pointCount[pillarCountIdx];
        if(pointNumInPillar > MAX_POINT_IN_PILLARS - 1)
            continue;


        //std::cout << "pillarsIndices2[" << pillarIdx << "]: " << pillarsIndices[pillarIdx] << std::endl;
        //std::cout << "pointCount2[" << pillarCountIdx << "]: " << pointCount[pillarCountIdx] << std::endl;

        feature[                                     pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = x;
        feature[1*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = y;
        feature[2*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = z; // z
        feature[3*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = points[idx*5+3]; // instence
        feature[4*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = points[idx*5+4]; // time_lag

        feature[8*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = x - (xIdx*X_STEP + X_MIN + X_STEP/2);
        feature[9*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarCountIdx*MAX_POINT_IN_PILLARS + pointNumInPillar] = y - (yIdx*Y_STEP + Y_MIN + Y_STEP/2);

        ++pointNumInPillar;
        pointCount[pillarCountIdx] = pointNumInPillar;
        
    }
    
    for(int pillarIdx = threadIdx*pillarsPerThread; pillarIdx < (threadIdx+1)*pillarsPerThread; pillarIdx++)
    {
        float xCenter = 0;
        float yCenter = 0;
        float zCenter = 0;
        auto pointNum = pointCount[pillarIdx];
        for(int pointIdx=0; pointIdx < pointNum; pointIdx++)
        {
            auto x = feature[                                     pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            auto y = feature[1*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            auto z = feature[2*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            xCenter += x;
            yCenter += y;
            zCenter += z;
        }
        xCenter = xCenter / pointNum;
        yCenter = yCenter / pointNum;
        zCenter = zCenter / pointNum;
        
        for(int pointIdx=0; pointIdx < pointNum; pointIdx++)
        {    
            auto x = feature[                                     pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            auto y = feature[1*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
            auto z = feature[2*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx];
       
            feature[5*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx] = x - xCenter;
            feature[6*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx] = y - yCenter;
            feature[7*MAX_PILLARS*MAX_POINT_IN_PILLARS + pillarIdx*MAX_POINT_IN_PILLARS + pointIdx] = z - zCenter;

        }
    }
    
}
