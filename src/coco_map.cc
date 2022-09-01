#include <iostream>
#include <functional>
#include <algorithm>

#include <json-c/json.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/gil/extension/io/jpeg_io.hpp>

#include <tkDNN/utils.h>

#include <tkDNN/evaluation.h>
#include "variable.h"

void convertFilename(std::string &filename,const std::string l_folder, const std::string i_folder, const std::string l_ext,const std::string i_ext)
{
    filename.replace(filename.find(l_folder),l_folder.length(),i_folder);
    filename.replace(filename.find(l_ext),l_ext.length(),i_ext);
}

static int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };

int get_index_from_coco_id(int classes, int id)
{
	int i = 0;
	for(i = 0 ; i < classes ; i++)
	{
		if(id == coco_ids[i])
			break;
	}
	return i;
}

void makeFrameMap(const char *labels_path, std::map<int, tk::dnn::Frame>& frame_map)
{
	std::string l_filename;
    std::ifstream all_labels(labels_path);

    for (int index=0 ; std::getline(all_labels, l_filename); ++index) {
		tk::dnn::Frame f;
        f.lFilename = l_filename;
        f.iFilename = l_filename;

        if(!fileExist(l_filename.c_str())) {
			std::string error_log = "label file " + l_filename + " is not exist";
        	FatalError(error_log);
		}

		convertFilename(f.iFilename, "labels", "images", ".txt", ".jpg");

		auto dims = boost::gil::jpeg_read_dimensions(f.iFilename);
		f.width = dims.x;
		f.height = dims.y;

		std::string id = l_filename.substr(l_filename.find("labels/")+7, l_filename.find(".txt") - l_filename.find("labels/") -7);
		int image_id = std::stoi(id);

		std::ifstream labels(l_filename);
		for(std::string line; std::getline(labels, line); ){
			std::istringstream in(line); 
			tk::dnn::BoundingBox b;
			in >> b.cl >> b.x >> b.y >> b.w >> b.h;  
			b.prob = 1;
			b.truthFlag = 1;
			f.gt.push_back(b);
		}

		frame_map.insert(std::pair<int, tk::dnn::Frame>(image_id, f));
	}
}


int main(int argc, char *argv[]) 
{
    int classes, map_points, map_levels;
    float map_step, IoU_thresh, conf_thresh;
    std::vector<tk::dnn::Frame> images;
    const char *config_filename = "../demo/config.yaml";
    const char * labels_path = "../demo/COCO_val2017/all_labels.txt";
	const char *coco_result_path = "results/coco_results.json";
	json_object * pJsonObject = NULL;
	bool verbose = true;

	if(argc < 4) {
		std::cerr<<"---- usage ----"<<std::endl;
		std::cerr<<"./compute_coco_map <labels_path> <yaml file> <results file>"<<std::endl;
		std::cerr<<"ex) ./compute_coco_map ./all_labels.txt ./config.yaml ./result.json"<<std::endl;
		FatalError("The number of arguments is less than four");	
	}

	labels_path = argv[1]; 
	config_filename = argv[2]; 
	coco_result_path = argv[3];

    //check if files needed exist
    if(!fileExist(config_filename))
        FatalError("Wrong config file path.");
    if(!fileExist(labels_path))
        FatalError("Wrong labels file path.");
    if(!fileExist(coco_result_path))
        FatalError("Wrong coco result file path.");

    // std::ifstream all_labels(labels_path);
    // std::string l_filename;
	std::map<int, tk::dnn::Frame> frame_map;
	int index = 0, coco_length = 0;

	// read label info
	makeFrameMap(labels_path, frame_map);

    //read mAP parameters
    tk::dnn::readmAPParams( config_filename, classes,  map_points, map_levels, map_step, IoU_thresh, conf_thresh, verbose);

	pJsonObject = json_object_from_file(coco_result_path);

	coco_length = json_object_array_length(pJsonObject);

	while(index < coco_length)
	{
		json_object *temp = json_object_array_get_idx(pJsonObject, index);
		json_object *imageIdObj = json_object_object_get(temp,"image_id");
		int image_id_json = json_object_get_int(imageIdObj);

		auto it = frame_map.find(image_id_json);
		if(it != frame_map.end()) {
			tk::dnn::Frame* f = &(it->second);

			tk::dnn::BoundingBox b;
			json_object *categoryIdObj = json_object_object_get(temp,"category_id");
			int category_id_json = json_object_get_int(categoryIdObj);

			json_object *scoreObj = json_object_object_get(temp,"score");
			float score_json = (float) json_object_get_double(scoreObj);
			json_object *bboxObj = json_object_object_get(temp,"bbox");

			b.x = (float) (json_object_get_double(json_object_array_get_idx(bboxObj,0)) + (float) json_object_get_int(json_object_array_get_idx(bboxObj,2))/2) / (float) f->width;
			b.y = (float) (json_object_get_double(json_object_array_get_idx(bboxObj,1)) + (float) json_object_get_int(json_object_array_get_idx(bboxObj,3))/2) / (float) f->height;
			b.w = (float) json_object_get_double(json_object_array_get_idx(bboxObj,2)) / (float) f->width;
			b.h = (float) json_object_get_double(json_object_array_get_idx(bboxObj,3)) / (float) f->height;
			//b.x = (d.x + d.w/2) / f.width;
			//b.y = (d.y + d.h/2) / f.height;
			//b.w = d.w / f.width;
			//b.h = d.h / f.height;

			b.prob = score_json;
			b.cl = get_index_from_coco_id(classes, category_id_json);
			//printf("id: %d, category: %d, x: %lf, y: %lf, w: %lf, h: %lf, prob: %f\n", image_id_json, category_id_json, b.x, b.y, b.w, b.h, b.prob);
			f->det.push_back(b);
		}
		
		index++;	
	}

	for(auto it = frame_map.begin(); it != frame_map.end(); it++) {
		images.push_back(it->second);
	}

    //compute mAP
    double AP = tk::dnn::computeMapNIoULevels(images,classes,IoU_thresh,conf_thresh, map_points, map_step, map_levels, verbose, false, "coco_trained_network");
    std::cout<<"mAP "<<IoU_thresh<<":"<<IoU_thresh+map_step*(map_levels-1)<<" = "<<AP<<std::endl;

    //compute average precision, recall and f1score
    tk::dnn::computeTPFPFN(images,classes,IoU_thresh,conf_thresh, verbose, false, "coco_trained_network");

	json_object_put(pJsonObject);
	pJsonObject = NULL;

    return 0;
}

