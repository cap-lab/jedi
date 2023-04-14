#include "config.h"
#include "variable.h"
#include "dataset.h"
#include "thread.h"

long getTime() {
	struct timespec time;
	if(0 != clock_gettime(CLOCK_REALTIME, &time)) {
		std::cerr<<"Something wrong on clock_gettime()"<<std::endl;		
		exit(-1);
	}
	return (time.tv_nsec) / 1000 + time.tv_sec * 1000000;
}


bool fileExist(std::string fname) {
    std::ifstream dataFile (fname.c_str(), std::ios::in | std::ios::binary);
    if(!dataFile)
    	return false;
    return true;
}


