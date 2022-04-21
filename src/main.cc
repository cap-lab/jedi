#include "runner.h"

int main(int argc, char *argv[]) {
	std::string config_file_name = "config.cfg";
	std::vector<IInferenceApplication *> apps;

	Runner runner(config_file_name, 416, 416, 3);

	runner.init();
	
	const char *filename = "tmp2.bin";
	float *input = (float *)malloc(sizeof(float) * 416 * 416 * 3);

    FILE *fp = fopen((char *)filename, "rb");
    fread(input, sizeof(float), 416*416*3, fp);
    fclose(fp);	

	std::cout<<"first image"<<std::endl;
	runner.run((char *)input, (char *)"image1.jpg");

    fp = fopen((char *)filename, "rb");
    fread(input, sizeof(float), 416*416*3, fp);
    fclose(fp);	

	std::cout<<"second image"<<std::endl;
	runner.run((char *)input, (char *)"image2.jpg");

	runner.wrapup();

	return 0;
}
