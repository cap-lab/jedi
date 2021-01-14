#include <iostream>
#include <fstream>
#include <vector>

#include <tkDNN/tkdnn.h>
#include <tkDNN/DarknetParser.h>
#include <tkDNN/Network.h>
#include <tkDNN/Layer.h>

#include "config.h"
#include "variable.h"
#include "model.h"
#include "dataset.h"
#include "thread.h"
#include "coco.h"

static void printHelpMessage() {
	std::cout<<"usage:"<<std::endl;
	std::cout<<"	./network_analyzer -c config_file [-o output_file]" <<std::endl;
}

static void printRouteShortCutRange(ConfigData *config_data, std::string output_file_name, int instance_id)
{
	std::string bin_path(config_data->instances.at(instance_id).bin_path);
    std::string wgs_path  = bin_path + "/layers";
    std::string cfg_path(config_data->instances.at(instance_id).cfg_path);
    std::string name_path(config_data->instances.at(instance_id).name_path);
	tk::dnn::Network *net = NULL;

	net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);
	std::ofstream writeFile(output_file_name.data());

	if(writeFile.is_open())
	{
		for(int i = 0 ; i < net->num_layers ; i++)
		{
			tk::dnn::Layer *l = net->layers[i];
			if(l->getLayerType() == tk::dnn::LAYER_SHORTCUT)
			{
				tk::dnn::Shortcut* shortcutLayer = (tk::dnn::Shortcut *) l;
				tk::dnn::Layer *backLayer = shortcutLayer->backLayer;
				//std::cout << backLayer->id << " " << l->id  << std::endl;
				writeFile << backLayer->id << ":" << l->id  << ":" << 2 << std::endl;
			}
			else if(l->getLayerType() == tk::dnn::LAYER_ROUTE)
			{
				tk::dnn::Route *routeLayer = (tk::dnn::Route *) l;
				for(int iter = 0; iter < routeLayer->layers_n; iter++) {
					tk::dnn::Layer *currLayer = routeLayer->layers[iter];
					if(l->id - currLayer->id > 1) 
					{
						//std::cout << currLayer->id << " " << l->id  << std::endl;
						writeFile << currLayer->id << ":" << l->id << ":" << routeLayer->layers_n  << std::endl;
					}
				}

			}
		}
		writeFile.close();
	}
}

int main(int argc, char *argv[]) {
	int option;
	std::string config_file_name = "config.cfg";
	std::string out_file_name = "range.log";

	if(argc == 1) {
		printHelpMessage();
		return 0;
	}

	while((option = getopt(argc, argv, "c:o:h")) != -1) {
		switch(option) {
			case 'c':
				config_file_name = std::string(optarg);	
				break;
			case 'o':
				out_file_name = std::string(optarg);
				break;
			case 'h':
				printHelpMessage();
				break;
		}	
	}

	// read configurations
	ConfigData config_data(config_file_name);

	printRouteShortCutRange(&config_data, out_file_name, 0);

	return 0;
}
