#ifndef _PROFILER_H_
#define _PROFILER_H__

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

#include "NvInfer.h"

extern std::vector<long> pre_time_vec, post_time_vec;

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

		void saveLayerTimes(const char *max_filename, const char *avg_filename, std::vector<std::vector<long>> dla_profile_vec)
		{
			bool dla_recorded = false;
			float avg_value = 0.0f, max_value = 0.0f;
			FILE *avg_fp = fopen(avg_filename, "w");
			FILE *max_fp = fopen(max_filename, "w");

			auto it2 = pre_time_vec.begin();
			it2++;
			avg_value = std::accumulate(it2, pre_time_vec.end(), 0.0) / (pre_time_vec.size());
			max_value = (float)(*std::max_element(it2, pre_time_vec.end()));
			fprintf(avg_fp,"cpu:%d\n", (int)(avg_value)); // usec
			fprintf(max_fp,"cpu:%d\n", (int)(max_value)); // usec

			for (size_t i = 0; i < profile_.size(); i++)
			{
				avg_value = 0.0f;
				max_value = 0.0f;

				if(!isMappedDLA(profile_[i].first)) {
					int n = (profile_[i].second).size();
					if(n > 1) {
						auto it = (profile_[i].second).begin();
						it++;
						avg_value = std::accumulate(it, (profile_[i].second).end(), 0.0) / (n-1);		
						max_value = (float)(*std::max_element(it, (profile_[i].second).end()));
					}

					fprintf(avg_fp,"%-60.60s:%d\n", profile_[i].first.c_str(), (int)(avg_value*1000)); // usec
					fprintf(max_fp,"%-60.60s:%d\n", profile_[i].first.c_str(), (int)(max_value*1000)); // usec
				}
				else {
					if(dla_recorded) {
						continue;
					}

					for(int iter = 0; iter < 2; iter++) {
						int n = (dla_profile_vec.at(iter)).size();	
						avg_value = 0.0f;
						max_value = 0.0f;

						if (n > 1) {
							auto it = (dla_profile_vec.at(iter)).begin();	
							it++;
							avg_value = std::accumulate(it, (dla_profile_vec.at(iter)).end(), 0.0) / (n-1);
							max_value = (float)(*std::max_element(it, (dla_profile_vec.at(iter)).end()));
						}

						if(avg_value > 0) {
							fprintf(avg_fp,"dla%d:%d\n", iter, (int)(avg_value)); // usec
						}
						if(max_value > 0) {
							fprintf(max_fp,"dla%d:%d\n", iter, (int)(max_value)); // usec
						}
					}
					dla_recorded = true;
				}
			}

			it2 = post_time_vec.begin();
			it2++;
			avg_value = std::accumulate(it2, post_time_vec.end(), 0.0) / (post_time_vec.size());
			max_value = (float)(*std::max_element(it2, post_time_vec.end()));
			fprintf(avg_fp,"cpu:%d\n", (int)(avg_value)); // usec
			fprintf(max_fp,"cpu:%d\n", (int)(max_value)); // usec

			fclose(avg_fp);
			fclose(max_fp);
		}

		bool isMappedDLA(std::string layer_name) {
			std::string nvm("nvm"); 
			std::string finish("finish"); 
			std::string comma(",");
		
			bool not_found = (layer_name.find(nvm) == std::string::npos) && (layer_name.find(finish) == std::string::npos) && (layer_name.find(comma) == std::string::npos);

			return !not_found;
		}

	protected:
		void reportLayerTime(const char *layerName, float ms) override
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
