#include "runner2.h"
// #include <opencv2/imgcodecs.hpp>


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

	std::string config_file_name(config_file);
	int w = 256, h = 256, c = 3, step = 768;
	// int ow = 0, oh = 0;
	int test_num = 1000;

	const char *filename = "tmp256.bin";
	char *input = (char *)malloc(w*h*c*sizeof(char));
	// float *input_buffer = new float[w * h * c];
	float **output_pointers = (float **) calloc(1, sizeof(float *));

	FILE *fp = fopen((char *)filename, "rb");
	fread(input, sizeof(char), w*h*c, fp);
	fclose(fp);	

	Runner2 runner2(config_file_name, w, h, c, step);

	runner2.init();

	/*
	for (int y = 0; y < h; ++y) {
		for (int k = 0; k < c; ++k) {
			for (int x = 0; x < w; ++x) {
				input_buffer[k*w*h + y*w + x] = input[y*step + x*c + k] / 255.0f;
			}
		}
	}
	runner2.setInputData(input_buffer);
	*/

	for(int iter = 0; iter < test_num; iter++) {
		runner2.setInputData2(input);

		runner2.runInference();

		runner2.getOutputData(output_pointers);

		runner2.doPostProcessing(output_pointers, (char *)input, (char *)"test-image.jpg");
		// runner2.doPostProcessing(output_pointers, nullptr, nullptr);
	}

	runner2.saveProfileResults(max_profile_file, avg_profile_file, min_profile_file);

	runner2.saveResults(result_file);

	runner2.wrapup();

	return 0;
}
