#ifndef _PROFILER_H_
#define _PROFILER_H__

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "NvInferRuntime.h"

extern std::vector<long> pre_time_vec, post_time_vec;

#define SIGMA 6

template<typename T>
T variance(const std::vector<T> &vec) {
	const size_t sz = vec.size();
	if (sz == 1) {
		return 0.0;
	}   

	// Calculate the mean
	const T mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz; 

	// Now calculate the variance
	auto variance_func = [&mean, &sz](T accumulator, const T& val) {
		return accumulator + ((val - mean)*(val - mean) / sz);
	};  

	return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
}

class Profiler : public nvinfer1::IProfiler
{
	public:
		void printLayerTimes()
		{   
			float total_time = 0;
			for (size_t i = 0; i < profile_.size(); i++)
			{   
				float value = 0.0f;
				int n = (profile_[i].second).size();

				if(n > 1) {
					auto it = (profile_[i].second).begin();
					it++;
					// value = std::accumulate(it, (profile_[i].second).end(), 0.0) / (n-1);        
					value = (float)(*std::max_element(it, (profile_[i].second).end()));
				}   

				printf("%d:%-60.60s:%d\n", (int)i, profile_[i].first.c_str(), (int)(value*1000)); // usec
				total_time += value;
			}   
			printf("All layers  : %d\n", (int)(total_time*1000));
		}   

		void saveLayerTimes(const char *max_filename, const char *avg_filename, const char *min_filename, std::vector<std::vector<long>> dla_profile_vec)
		{
			bool dla_recorded = false;
			float avg_value = 0.0f, max_value = 0.0f, max_value2 = 0.0f, var = 0.0f, min_value = 0.0f, min_value2 = 0.0f;
			FILE *avg_fp = fopen(avg_filename, "w");
			FILE *max_fp = fopen(max_filename, "w");
			FILE *min_fp = fopen(min_filename, "w");

			auto it2 = pre_time_vec.begin();
			avg_value = std::accumulate(it2, pre_time_vec.end(), 0.0) / (pre_time_vec.size());
			max_value2 = (float)(*std::max_element(it2, pre_time_vec.end()));
			min_value2 = (float)(*std::min_element(it2, pre_time_vec.end()));
			var = variance(pre_time_vec);
			max_value = avg_value + SIGMA * sqrt(var);
			// max_value = max_value > max_value2 ? max_value : max_value2;
			min_value = min_value2;
			// min_value = avg_value - SIGMA * sqrt(var);
			// min_value = min_value < min_value2 ? min_value : min_value2;
			// min_value = min_value < 0 ? 0 : min_value;

			fprintf(avg_fp,"cpu:%d\n", (int)(avg_value)); // usec
			fprintf(max_fp,"cpu:%d\n", (int)(max_value)); // usec
			fprintf(min_fp,"cpu:%d\n", (int)(min_value)); // usec

			for (size_t i = 0; i < profile_.size(); i++)
			{
				avg_value = 0.0f;
				max_value = 0.0f;
				min_value = 0.0f;

				if(!isMappedDLA(profile_[i].first)) {
					int n = (profile_[i].second).size();
					if(n > 1) {
						auto it = (profile_[i].second).begin();
						avg_value = std::accumulate(it, (profile_[i].second).end(), 0.0) / (n-1);
						max_value2 = (float)(*std::max_element(it, (profile_[i].second).end()));
						min_value2 = (float)(*std::min_element(it, (profile_[i].second).end()));
						var = variance(profile_[i].second);
						max_value = avg_value + SIGMA * sqrt(var);
						// max_value = max_value > max_value2 ? max_value : max_value2;
						min_value = min_value2;
						// min_value = avg_value - SIGMA * sqrt(var);
						// min_value = min_value < min_value2 ? min_value : min_value2;
						// min_value = min_value < 0 ? 0 : min_value;
					}

					fprintf(avg_fp,"%-60.60s:%d\n", profile_[i].first.c_str(), (int)(avg_value*1000)); // usec
					fprintf(max_fp,"%-60.60s:%d\n", profile_[i].first.c_str(), (int)(max_value*1000)); // usec
					fprintf(min_fp,"%-60.60s:%d\n", profile_[i].first.c_str(), (int)(min_value*1000)); // usec
				}
				else {
					if(dla_recorded) {
						continue;
					}

					for(int iter = 0; iter < 2; iter++) {
						int n = (dla_profile_vec.at(iter)).size();
						avg_value = 0.0f;
						max_value = 0.0f;
						min_value = 0.0f;

						if (n > 1) {
							auto it = (dla_profile_vec.at(iter)).begin();
							avg_value = std::accumulate(it, (dla_profile_vec.at(iter)).end(), 0.0) / (n-1);
							max_value2 = (float)(*std::max_element(it, (dla_profile_vec.at(iter)).end()));
							min_value2 = (float)(*std::min_element(it, (dla_profile_vec.at(iter)).end()));
							var = variance(dla_profile_vec.at(iter));
							max_value = avg_value + SIGMA * sqrt(var);
							// max_value = max_value > max_value2 ? max_value : max_value2;
							min_value = min_value2;
							// min_value = avg_value - SIGMA * sqrt(var);
							// min_value = min_value < min_value2 ? min_value : min_value2;
							// min_value = min_value < 0 ? 0 : min_value;
						}

						if(avg_value > 0) {
							fprintf(avg_fp,"dla%d:%d\n", iter, (int)(avg_value)); // usec
						}
						if(max_value > 0) {
							fprintf(max_fp,"dla%d:%d\n", iter, (int)(max_value)); // usec
						}
						if(min_value > 0) {
							fprintf(min_fp,"dla%d:%d\n", iter, (int)(min_value)); // usec
						}
					}
					dla_recorded = true;
				}
			}

			it2 = post_time_vec.begin();
			avg_value = std::accumulate(it2, post_time_vec.end(), 0.0) / (post_time_vec.size());
			max_value2 = (float)(*std::max_element(it2, post_time_vec.end()));
			min_value2 = (float)(*std::min_element(it2, post_time_vec.end()));
			var = variance(post_time_vec);
			max_value = avg_value + SIGMA * sqrt(var);
			// max_value = max_value > max_value2 ? max_value : max_value2;
			min_value = min_value2;
			// min_value = avg_value - SIGMA * sqrt(var);
			// min_value = min_value < min_value2 ? min_value : min_value2;
			// min_value = min_value < 0 ? 0 : min_value;
			fprintf(avg_fp,"cpu:%d\n", (int)(avg_value)); // usec
			fprintf(max_fp,"cpu:%d\n", (int)(max_value)); // usec
			fprintf(min_fp,"cpu:%d\n", (int)(min_value)); // usec

			fclose(avg_fp);
			fclose(max_fp);
			fclose(min_fp);
		}

		bool isMappedDLA(std::string layer_name) {
			std::string nvm("nvm");
			std::string finish("finish");
			std::string foreign("Foreign");

			bool not_found = (layer_name.find(nvm) == std::string::npos) && (layer_name.find(finish) == std::string::npos) && (layer_name.find(foreign) == std::string::npos);

			return !not_found;
		}

	protected:
		void reportLayerTime(const char *layerName, float ms) noexcept override
		{
			auto record = std::find_if(profile_.begin(), profile_.end(), [&](const Record &r) { return r.first == layerName; });
			if (record == profile_.end()) {
				std::vector<float> vec;
				vec.push_back(ms);

				profile_.push_back(std::make_pair(layerName, vec));
			}
			else {
				(record->second).push_back(ms);
				// record->second = std::max(record->second, ms);
				// record->second = ms;
			}
			// std::cerr<<layerName<<": "<<(int)(ms*1000)<<std::endl;
		}

	private:
		using Record = std::pair<std::string, std::vector<float>>;
		std::vector<Record> profile_;
};

#endif
