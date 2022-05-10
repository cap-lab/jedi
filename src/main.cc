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

	std::string config_file_name(argv[1]);
	std::vector<IInferenceApplication *> apps;

	Runner runner(config_file_name, 256, 256, 3, 768);

	runner.init();

	const char *filename = "tmp256.bin";
	char *input = (char *)malloc(256*256*3*sizeof(char));

	FILE *fp = fopen((char *)filename, "rb");
	fread(input, sizeof(char), 256*256*3, fp);
	fclose(fp);	

	runner.run((char *)input, (char *)"test-image.jpg");

	// runner.saveProfileResults(max_profile_file, avg_profile_file, min_profile_file);

	runner.wrapup();

	return 0;
}
