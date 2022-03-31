#include "runner.h"

int main(int argc, char *argv[]) {
	std::string config_file_name = "config.cfg";
	std::vector<IInferenceApplication *> apps;

	Runner runner(config_file_name);

	runner.init();

	// const char *path = "/sdcard/data/coco2017/images/000000283785.jpg";
	// runner.run_with_path((char *)path);
	
	const char *filename = "tmp.bin";
    FILE *fp = fopen((char *)filename, "rb");
	float *input = (float *)malloc(sizeof(float) * 416 * 416 * 3 * 4);
    fread(input, sizeof(float), 416*416*3*4, fp);
    fclose(fp);	

	std::cout<<"first image"<<std::endl;
	runner.run_with_data((char *)input, 416, 416, 3);

	std::cout<<"second image"<<std::endl;
	runner.run_with_data((char *)input, 416, 416, 3);

	runner.wrapup();

	return 0;
}
