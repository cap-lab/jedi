#ifndef _RESULT_FORMAT_H_
#define _RESULT_FORMAT_H_

#include <iostream>
#include <string>
#include <list>

#include "box.h"
#include "config.h"
#include "variable.h"
#include "image_dataset.h"

class ResultFormat {
	public:
		ResultFormat() {};
		virtual ~ResultFormat() {};

		virtual void writeResultFile(std::string result_file_name) = 0;
};

#endif
