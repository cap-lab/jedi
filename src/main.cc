#include "runner.h"

int main(int argc, char *argv[]) {
	std::string config_file_name = "config.cfg";
	std::vector<IInferenceApplication *> apps;

	Runner runner(config_file_name, 416, 416, 3, 1248);

	runner.init();
	
	const char *filename = "tmp.bin";
	char *input = (char *)malloc(416 * 416 * 3);

    FILE *fp = fopen((char *)filename, "rb");
    fread(input, sizeof(char), 416*416*3, fp);
    fclose(fp);	

	std::cout<<"first image"<<std::endl;
	runner.run((char *)input, (char *)"image1.jpg");

    fp = fopen((char *)filename, "rb");
    fread(input, sizeof(char), 416*416*3, fp);
    fclose(fp);	

	std::cout<<"second image"<<std::endl;
	runner.run((char *)input, (char *)"image2.jpg");

	runner.saveProfileResults("max.log", "avg.log", "min.log");

	runner.wrapup();

	return 0;
}
