#include "runner.h"

int main(int argc, char *argv[]) {
	if(argc < 5) {
		printf("arguments are not enough\n");
		return 0;
	}
	char *config_file = argv[1];
	char *max_profile_file = argv[2];
	char *avg_profile_file = argv[3];
	char *min_profile_file = argv[4];
	char *result_file = argv[5];

	std::string config_file_name(argv[1]);
	int w = 256, h = 256, c = 3, step = 768;
	// int ow = 0, oh = 0;
	int test_num = 1000;

	const char *filename = "tmp256.bin";
	char *input = (char *)malloc(w*h*c*sizeof(char));

	FILE *fp = fopen((char *)filename, "rb");
	fread(input, sizeof(char), w*h*c, fp);
	fclose(fp);	

	Runner runner(config_file_name, w, h, c, step);

	runner.init();

	for(int iter = 0; iter < test_num; iter++) {
		runner.run((char *)input, (char *)"test-image.jpg");
		// runner.run((char *)input, nullptr);
	}

	runner.saveProfileResults(max_profile_file, avg_profile_file, min_profile_file);

	runner.saveResults(result_file);

	runner.wrapup();

	return 0;
}
