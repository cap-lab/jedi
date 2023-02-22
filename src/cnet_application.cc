#include <libconfig.h++>
#include <cstring>
#include <sstream>

#include "variable.h"
#include "box.h"
#include "coco.h"

#include "image_opencv_cnet.h"

#include "cnet_detections.h"

#include "cnet_application.h"

#include "tkdnn_network.h"

const std::string conv1_bin = "base-base_layer-0.bin";
const std::string conv2_bin = "base-level0-0.bin";
const std::string conv3_bin = "base-level1-0.bin";
// s - stage, t - tree
const std::string s1_t1_conv1_bin = "base-level2-tree1-conv1.bin";
const std::string s1_t1_conv2_bin = "base-level2-tree1-conv2.bin";
const std::string s1_t1_project = "base-level2-project-0.bin";
const std::string s1_t2_conv1_bin = "base-level2-tree2-conv1.bin";
const std::string s1_t2_conv2_bin = "base-level2-tree2-conv2.bin";
const std::string s1_root_conv1_bin = "base-level2-root-conv.bin";
const std::string s2_t1_t1_conv1_bin = "base-level3-tree1-tree1-conv1.bin";
const std::string s2_t1_t1_conv2_bin = "base-level3-tree1-tree1-conv2.bin";
const std::string s2_t1_t1_project = "base-level3-tree1-project-0.bin";
const std::string s2_t1_t2_conv1_bin = "base-level3-tree1-tree2-conv1.bin";
const std::string s2_t1_t2_conv2_bin = "base-level3-tree1-tree2-conv2.bin";
const std::string s2_t1_root_conv1_bin = "base-level3-tree1-root-conv.bin";
const std::string s2_t2_t1_conv1_bin = "base-level3-tree2-tree1-conv1.bin";
const std::string s2_t2_t1_conv2_bin = "base-level3-tree2-tree1-conv2.bin";
const std::string s2_t2_t2_conv1_bin = "base-level3-tree2-tree2-conv1.bin";
const std::string s2_t2_t2_conv2_bin = "base-level3-tree2-tree2-conv2.bin";
const std::string s2_t2_root_conv1_bin = "base-level3-tree2-root-conv.bin";
const std::string s3_t1_t1_conv1_bin = "base-level4-tree1-tree1-conv1.bin";
const std::string s3_t1_t1_conv2_bin = "base-level4-tree1-tree1-conv2.bin";
const std::string s3_t1_t1_project = "base-level4-tree1-project-0.bin";
const std::string s3_t1_t2_conv1_bin = "base-level4-tree1-tree2-conv1.bin";
const std::string s3_t1_t2_conv2_bin = "base-level4-tree1-tree2-conv2.bin";
const std::string s3_t1_root_conv1_bin = "base-level4-tree1-root-conv.bin";
const std::string s3_t2_t1_conv1_bin = "base-level4-tree2-tree1-conv1.bin";
const std::string s3_t2_t1_conv2_bin = "base-level4-tree2-tree1-conv2.bin";
const std::string s3_t2_t2_conv1_bin = "base-level4-tree2-tree2-conv1.bin";
const std::string s3_t2_t2_conv2_bin = "base-level4-tree2-tree2-conv2.bin";
const std::string s3_t2_root_conv1_bin = "base-level4-tree2-root-conv.bin";
const std::string s4_t1_conv1_bin = "base-level5-tree1-conv1.bin";
const std::string s4_t1_conv2_bin = "base-level5-tree1-conv2.bin";
const std::string s4_t1_project = "base-level5-project-0.bin";
const std::string s4_t2_conv1_bin = "base-level5-tree2-conv1.bin";
const std::string s4_t2_conv2_bin = "base-level5-tree2-conv2.bin";
const std::string s4_root_conv1_bin = "base-level5-root-conv.bin";

//final
// const std::string fc_bin = "output.bin";

const std::string ida_0_p_1_dcn_bin = "dla_up-ida_0-proj_1-conv.bin";
const std::string ida_0_p_1_conv_bin = "dla_up-ida_0-proj_1-conv-conv_offset_mask.bin";
const std::string ida_0_up_1_deconv_bin = "dla_up-ida_0-up_1.bin";
const std::string ida_0_n_1_dcn_bin = "dla_up-ida_0-node_1-conv.bin";
const std::string ida_0_n_1_conv_bin = "dla_up-ida_0-node_1-conv-conv_offset_mask.bin";

const std::string ida_1_p_1_dcn_bin = "dla_up-ida_1-proj_1-conv.bin";
const std::string ida_1_p_1_conv_bin = "dla_up-ida_1-proj_1-conv-conv_offset_mask.bin";
const std::string ida_1_up_1_deconv_bin = "dla_up-ida_1-up_1.bin";
const std::string ida_1_n_1_dcn_bin = "dla_up-ida_1-node_1-conv.bin";
const std::string ida_1_n_1_conv_bin = "dla_up-ida_1-node_1-conv-conv_offset_mask.bin";
const std::string ida_1_p_2_dcn_bin = "dla_up-ida_1-proj_2-conv.bin";
const std::string ida_1_p_2_conv_bin = "dla_up-ida_1-proj_2-conv-conv_offset_mask.bin";
const std::string ida_1_up_2_deconv_bin = "dla_up-ida_1-up_2.bin";
const std::string ida_1_n_2_dcn_bin = "dla_up-ida_1-node_2-conv.bin";
const std::string ida_1_n_2_conv_bin = "dla_up-ida_1-node_2-conv-conv_offset_mask.bin";

const std::string ida_2_p_1_dcn_bin = "dla_up-ida_2-proj_1-conv.bin";
const std::string ida_2_p_1_conv_bin = "dla_up-ida_2-proj_1-conv-conv_offset_mask.bin";
const std::string ida_2_up_1_deconv_bin = "dla_up-ida_2-up_1.bin";
const std::string ida_2_n_1_dcn_bin = "dla_up-ida_2-node_1-conv.bin";
const std::string ida_2_n_1_conv_bin = "dla_up-ida_2-node_1-conv-conv_offset_mask.bin";
const std::string ida_2_p_2_dcn_bin = "dla_up-ida_2-proj_2-conv.bin";
const std::string ida_2_p_2_conv_bin = "dla_up-ida_2-proj_2-conv-conv_offset_mask.bin";
const std::string ida_2_up_2_deconv_bin = "dla_up-ida_2-up_2.bin";
const std::string ida_2_n_2_dcn_bin = "dla_up-ida_2-node_2-conv.bin";
const std::string ida_2_n_2_conv_bin = "dla_up-ida_2-node_2-conv-conv_offset_mask.bin";
const std::string ida_2_p_3_dcn_bin = "dla_up-ida_2-proj_3-conv.bin";
const std::string ida_2_p_3_conv_bin = "dla_up-ida_2-proj_3-conv-conv_offset_mask.bin";
const std::string ida_2_up_3_deconv_bin = "dla_up-ida_2-up_3.bin";
const std::string ida_2_n_3_dcn_bin = "dla_up-ida_2-node_3-conv.bin";
const std::string ida_2_n_3_conv_bin = "dla_up-ida_2-node_3-conv-conv_offset_mask.bin";

const std::string ida_up_p_1_dcn_bin = "ida_up-proj_1-conv.bin";
const std::string ida_up_p_1_conv_bin = "ida_up-proj_1-conv-conv_offset_mask.bin";
const std::string ida_up_up_1_deconv_bin = "ida_up-up_1.bin";
const std::string ida_up_n_1_dcn_bin = "ida_up-node_1-conv.bin";
const std::string ida_up_n_1_conv_bin = "ida_up-node_1-conv-conv_offset_mask.bin";
const std::string ida_up_p_2_dcn_bin = "ida_up-proj_2-conv.bin";
const std::string ida_up_p_2_conv_bin = "ida_up-proj_2-conv-conv_offset_mask.bin";
const std::string ida_up_up_2_deconv_bin = "ida_up-up_2.bin";
const std::string ida_up_n_2_dcn_bin = "ida_up-node_2-conv.bin";
const std::string ida_up_n_2_conv_bin = "ida_up-node_2-conv-conv_offset_mask.bin";

const std::string hm_conv1_bin = "hm-0.bin";
const std::string hm_conv2_bin = "hm-2.bin";
const std::string wh_conv1_bin = "wh-0.bin";
const std::string wh_conv2_bin = "wh-2.bin";
const std::string reg_conv1_bin = "reg-0.bin";
const std::string reg_conv2_bin = "reg-2.bin";

REGISTER_JEDI_APPLICATION(CenternetApplication);

void CenternetApplication::readImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		centernetAppConfig.image_path = data.c_str();

		std::cerr<<"image_path: "<<centernetAppConfig.image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'image_path' setting in configuration file." << std::endl;
	}
}

void CenternetApplication::readCalibImagePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["calib_image_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;
		centernetAppConfig.calib_image_path = data.c_str();

		std::cerr<<"calib_image_path: "<<centernetAppConfig.calib_image_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_image_path' setting in configuration file." << std::endl;
	}
}

void CenternetApplication::readCalibImagesNum(libconfig::Setting &setting){
	try {
		const char *data = setting["calib_images_num"];
		centernetAppConfig.calib_images_num = atoi(data);

		std::cerr<<"calib_images_num: "<<centernetAppConfig.calib_images_num<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'calib_images_num' setting in configuration file." <<std::endl;
	}
}

void CenternetApplication::readNamePath(libconfig::Setting &setting) {
	try{	
		const char *tmp = setting["name_path"];
		std::stringstream ss(tmp);
		static std::string data;
		ss >> data;

		centernetAppConfig.name_path = data.c_str();

		std::cerr<<"name_path: "<<centernetAppConfig.name_path<<std::endl;
	}
	catch(const libconfig::SettingNotFoundException &nfex) {
		std::cerr << "No 'name_path' setting in configuration file." << std::endl;
	}
}


void CenternetApplication::readCustomOptions(libconfig::Setting &setting)
{
	readImagePath(setting);
	readCalibImagePath(setting);
	readCalibImagesNum(setting);
	readNamePath(setting);
}

IJediNetwork *CenternetApplication::createNetwork(ConfigInstance *basic_config_data)
{
	std::string name_path = centernetAppConfig.name_path;
	std::string bin_path(basic_config_data->bin_path);
    std::string wgs_path  = bin_path + "/layers/";
	TkdnnNetwork *jedi_network = new TkdnnNetwork();

	tk::dnn::dataDim_t dim(1, 3, 512, 512, 1);
    tk::dnn::Network *net = new tk::dnn::Network(dim);
    tk::dnn::Layer *last1, *last2, *last3, *last4;
    tk::dnn::Layer *base2, *base3, *base4, *base5, *ida1, *ida2_1, *ida2_2, *ida3_1, *ida3_2, *ida3_3, *idaup_1, *idaup_2;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    tk::dnn::Conv2d *conv1 = new tk::dnn::Conv2d(net, 16, 7, 7, 1, 1, 3, 3, (wgs_path + conv1_bin).c_str(), true);
    tk::dnn::Activation *relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d *conv2 = new tk::dnn::Conv2d(net, 16, 3, 3, 1, 1, 1, 1, (wgs_path + conv2_bin).c_str(), true);
    tk::dnn::Activation *relu2 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    //base1 = relu2;

    tk::dnn::Conv2d *conv3 = new tk::dnn::Conv2d(net, 32, 3, 3, 2, 2, 1, 1, (wgs_path + conv3_bin).c_str(), true);
    tk::dnn::Activation *relu3 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    base2 = relu3;
    
    // level 2
    // tree 1
        tk::dnn::Conv2d     *s1_t1_conv1 = new tk::dnn::Conv2d(net, 64, 3, 3, 2, 2, 1, 1, (wgs_path + s1_t1_conv1_bin).c_str(), true);
        tk::dnn::Activation *s1_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d     *s1_t1_conv2 = new tk::dnn::Conv2d(net, 64, 3, 3, 1, 1, 1, 1, (wgs_path + s1_t1_conv2_bin).c_str(), true);
        last2 = s1_t1_conv2;

        // get the basicblock input and apply maxpool conv2d and relu
        tk::dnn::Layer      *route_s1_t1_layers[1] = { base2 };
        tk::dnn::Route      *route_s1_t1 = new tk::dnn::Route(net, route_s1_t1_layers, 1);
        // downsample
        tk::dnn::Pooling   *s1_t1_maxpool = new tk::dnn::Pooling(net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
        // project
        tk::dnn::Conv2d    *s1_t1_residual1_conv1 = new tk::dnn::Conv2d(net, 64, 1, 1, 1, 1, 0, 0, (wgs_path + s1_t1_project).c_str(), true);
        
        tk::dnn::Shortcut  *s1_t1_s1 = new tk::dnn::Shortcut(net, last2);
        tk::dnn::Activation *s1_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    
    last1 = s1_t1_relu;
    // tree 2
        tk::dnn::Conv2d     *s1_t2_conv1 = new tk::dnn::Conv2d(net, 64, 3, 3, 1, 1, 1, 1, (wgs_path + s1_t2_conv1_bin).c_str(), true);
        tk::dnn::Activation *s1_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d     *s1_t2_conv2 = new tk::dnn::Conv2d(net, 64, 3, 3, 1, 1, 1, 1, (wgs_path + s1_t2_conv2_bin).c_str(), true);
        
        tk::dnn::Shortcut   *s1_t2_s1 = new tk::dnn::Shortcut(net, last1);
        tk::dnn::Activation *s1_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
        last2 = s1_t2_relu;

    // root
        // join last1 and net in single input 128, 56, 56
        tk::dnn::Layer      *route_s1_root_layers[2] = { last2, last1 };
        tk::dnn::Route      *route_s1_root = new tk::dnn::Route(net, route_s1_root_layers, 2);
        tk::dnn::Conv2d     *s1_root_conv1 = new tk::dnn::Conv2d(net, 64, 1, 1, 1, 1, 0, 0, (wgs_path + s1_root_conv1_bin).c_str(), true);
        tk::dnn::Activation *s1_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);

    base3 = s1_root_relu;

    // level 3
    // tree 1
        // tree 1
            tk::dnn::Conv2d     *s2_t1_t1_conv1 = new tk::dnn::Conv2d(net, 128, 3, 3, 2, 2, 1, 1, (wgs_path + s2_t1_t1_conv1_bin).c_str(), true);
            tk::dnn::Activation *s2_t1_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);      
        
            tk::dnn::Conv2d     *s2_t1_t1_conv2 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t1_t1_conv2_bin).c_str(), true);
            last2 = s2_t1_t1_conv2;

            // get the basicblock input and apply maxpool conv2d and relu
            tk::dnn::Layer      *route_s2_t1_t1_layers[1] = { base3 };
            tk::dnn::Route      *route_s2_t1_t1 = new tk::dnn::Route(net, route_s2_t1_t1_layers, 1);
            // downsample
            tk::dnn::Pooling    *s2_t1_t1_maxpool1 = new tk::dnn::Pooling(net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
            last4 = s2_t1_t1_maxpool1;
            // project
            tk::dnn::Conv2d     *s2_t1_t1_residual1_conv1 = new tk::dnn::Conv2d(net, 128, 1, 1, 1, 1, 0, 0, (wgs_path + s2_t1_t1_project).c_str(), true);
            
            tk::dnn::Shortcut   *s2_t1_t1_s1 = new tk::dnn::Shortcut(net, last2);
            tk::dnn::Activation *s2_t1_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);

        last1 = s2_t1_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d     *s2_t1_t2_conv1 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t1_t2_conv1_bin).c_str(), true);
            tk::dnn::Activation *s2_t1_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     *s2_t1_t2_conv2 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t1_t2_conv2_bin).c_str(), true);
            
            tk::dnn::Shortcut   *s2_t1_t2_s1 = new tk::dnn::Shortcut(net, last1);
            tk::dnn::Activation *s2_t1_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
            last2 = s2_t1_t2_relu;

        // root
            // join last1 and net in single input 128, 56, 56
            tk::dnn::Layer      *route_s2_t1_root_layers[2] = { last2, last1 };
            tk::dnn::Route      *route_s2_t1_root = new tk::dnn::Route(net, route_s2_t1_root_layers, 2);
            tk::dnn::Conv2d     *s2_t1_root_conv1 = new tk::dnn::Conv2d(net, 128, 1, 1, 1, 1, 0, 0, (wgs_path + s2_t1_root_conv1_bin).c_str(), true);
            tk::dnn::Activation *s2_t1_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    
    last1 = s2_t1_root_relu;
    last3 = s2_t1_root_relu;
    // tree 2
        // tree 1
            tk::dnn::Conv2d     *s2_t2_t1_conv1 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t2_t1_conv1_bin).c_str(), true);
            tk::dnn::Activation *s2_t2_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     *s2_t2_t1_conv2 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t2_t1_conv2_bin).c_str(), true);
            tk::dnn::Shortcut   *s2_t2_t1_s1 = new tk::dnn::Shortcut(net, last1);
            tk::dnn::Activation *s2_t2_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);

        last1 = s2_t2_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d     *s2_t2_t2_conv1 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t2_t2_conv1_bin).c_str(), true);
            tk::dnn::Activation *s2_t2_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     *s2_t2_t2_conv2 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t2_t2_conv2_bin).c_str(), true);
            
            tk::dnn::Shortcut   *s2_t2_t2_s1 = new tk::dnn::Shortcut(net, last1);
            tk::dnn::Activation *s2_t2_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
            last2 = s2_t2_t2_relu;

        // root
            // join last1 and net in single input 128, 56, 56
            tk::dnn::Layer      *route_s2_t2_root_layers[4] = { last2, last1, last4, last3};
            tk::dnn::Route      *route_s2_t2_root = new tk::dnn::Route(net, route_s2_t2_root_layers, 4);
            tk::dnn::Conv2d     *s2_t2_root_conv1 = new tk::dnn::Conv2d(net, 128, 1, 1, 1, 1, 0, 0, (wgs_path + s2_t2_root_conv1_bin).c_str(), true);
            tk::dnn::Activation *s2_t2_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    
    base4 = s2_t2_root_relu;

    // level 4
    // tree 1
        // tree 1
            tk::dnn::Conv2d     *s3_t1_t1_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 2, 2, 1, 1, (wgs_path + s3_t1_t1_conv1_bin).c_str(), true);
            tk::dnn::Activation *s3_t1_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);      
        
            tk::dnn::Conv2d     *s3_t1_t1_conv2 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t1_t1_conv2_bin).c_str(), true);
            last2 = s3_t1_t1_conv2;

            // get the basicblock input and apply maxpool conv2d and relu
            tk::dnn::Layer      *route_s3_t1_t1_layers[1] = { base4 };
            tk::dnn::Route      *route_s3_t1_t1 = new tk::dnn::Route(net, route_s3_t1_t1_layers, 1);
            // downsample
            tk::dnn::Pooling    *s3_t1_t1_maxpool1 = new tk::dnn::Pooling(net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
            last4 = s3_t1_t1_maxpool1;
            // project
            tk::dnn::Conv2d     *s3_t1_t1_residual1_conv1 = new tk::dnn::Conv2d(net, 256, 1, 1, 1, 1, 0, 0, (wgs_path + s3_t1_t1_project).c_str(), true);
            
            tk::dnn::Shortcut   *s3_t1_t1_s1 = new tk::dnn::Shortcut(net, last2);
            tk::dnn::Activation *s3_t1_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);

        last1 = s3_t1_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d     *s3_t1_t2_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t1_t2_conv1_bin).c_str(), true);
            tk::dnn::Activation *s3_t1_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     *s3_t1_t2_conv2 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t1_t2_conv2_bin).c_str(), true);
            
            tk::dnn::Shortcut   *s3_t1_t2_s1 = new tk::dnn::Shortcut(net, last1);
            tk::dnn::Activation *s3_t1_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
            last2 = s3_t1_t2_relu;

        // root
            // join last1 and net in single input 256, 56, 56
            tk::dnn::Layer      *route_s3_t1_root_layers[2] = { last2, last1 };
            tk::dnn::Route      *route_s3_t1_root = new tk::dnn::Route(net, route_s3_t1_root_layers, 2);
            tk::dnn::Conv2d     *s3_t1_root_conv1 = new tk::dnn::Conv2d(net, 256, 1, 1, 1, 1, 0, 0, (wgs_path + s3_t1_root_conv1_bin).c_str(), true);
            tk::dnn::Activation *s3_t1_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    
    last1 = s3_t1_root_relu;
    last3 = s3_t1_root_relu;
    // tree 2
        // tree 1
            tk::dnn::Conv2d     *s3_t2_t1_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t2_t1_conv1_bin).c_str(), true);
            tk::dnn::Activation *s3_t2_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     *s3_t2_t1_conv2 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t2_t1_conv2_bin).c_str(), true);
            tk::dnn::Shortcut   *s3_t2_t1_s1 = new tk::dnn::Shortcut(net, last1);
            tk::dnn::Activation *s3_t2_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);

        last1 = s3_t2_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d     *s3_t2_t2_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t2_t2_conv1_bin).c_str(), true);
            tk::dnn::Activation *s3_t2_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     *s3_t2_t2_conv2 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t2_t2_conv2_bin).c_str(), true);
            
            tk::dnn::Shortcut   *s3_t2_t2_s1 = new tk::dnn::Shortcut(net, last1);
            tk::dnn::Activation *s3_t2_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
            last2 = s3_t2_t2_relu;

        // root
            // join last1 and net in single input 256, 56, 56
            tk::dnn::Layer      *route_s3_t2_root_layers[4] = { last2, last1, last4, last3};
            tk::dnn::Route      *route_s3_t2_root = new tk::dnn::Route(net, route_s3_t2_root_layers, 4);
            tk::dnn::Conv2d     *s3_t2_root_conv1 = new tk::dnn::Conv2d(net, 256, 1, 1, 1, 1, 0, 0, (wgs_path + s3_t2_root_conv1_bin).c_str(), true);
            tk::dnn::Activation *s3_t2_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    
    base5 = s3_t2_root_relu;

    // level 5
    // tree 1
        tk::dnn::Conv2d     *s4_t1_conv1 = new tk::dnn::Conv2d(net, 512, 3, 3, 2, 2, 1, 1, (wgs_path + s4_t1_conv1_bin).c_str(), true);
        tk::dnn::Activation *s4_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d     *s4_t1_conv2 = new tk::dnn::Conv2d(net, 512, 3, 3, 1, 1, 1, 1, (wgs_path + s4_t1_conv2_bin).c_str(), true);
        last2 = s4_t1_conv2;

        // get the basicblock input and apply maxpool conv2d and relu
        tk::dnn::Layer      *route_s4_t1_layers[1] = { base5 };
        tk::dnn::Route      *route_s4_t1 = new tk::dnn::Route(net, route_s4_t1_layers, 1);
        // downsample
        tk::dnn::Pooling    *s4_t1_maxpool1 = new tk::dnn::Pooling(net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
        last4 = s4_t1_maxpool1;
        // project
        tk::dnn::Conv2d     *s4_t1_residual1_conv1 = new tk::dnn::Conv2d(net, 512, 1, 1, 1, 1, 0, 0, (wgs_path + s4_t1_project).c_str(), true);
        
        tk::dnn::Shortcut   *s4_t1_s1 = new tk::dnn::Shortcut(net, last2);
        tk::dnn::Activation *s4_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);

    last1 = s4_t1_relu;
    
    // tree 2
        tk::dnn::Conv2d     *s4_t2_conv1 = new tk::dnn::Conv2d(net, 512, 3, 3, 1, 1, 1, 1, (wgs_path + s4_t2_conv1_bin).c_str(), true);
        tk::dnn::Activation *s4_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d     *s4_t2_conv2 = new tk::dnn::Conv2d(net, 512, 3, 3, 1, 1, 1, 1, (wgs_path + s4_t2_conv2_bin).c_str(), true);
        
        tk::dnn::Shortcut   *s4_t2_s1 = new tk::dnn::Shortcut(net, last1);
        tk::dnn::Activation *s4_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
        last2 = s4_t2_relu;

    // root
        // join last1 and net in single input 128, 56, 56
        tk::dnn::Layer      *route_s4_root_layers[3] = { last2, last1, last4 };
        tk::dnn::Route      *route_s4_root = new tk::dnn::Route(net, route_s4_root_layers, 3);
        tk::dnn::Conv2d     *s4_root_conv1 = new tk::dnn::Conv2d(net, 512, 1, 1, 1, 1, 0, 0, (wgs_path + s4_root_conv1_bin).c_str(), true);
        tk::dnn::Activation *s4_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);

    //base6 = s4_root_relu;
    
    //final
    // tk::dnn::Pooling avgpool(net, 7, 7, 7, 7, 0, 0, tk::dnn::POOLING_AVERAGE);
    // tk::dnn::Dense   fc(net, 1000, fc_bin);
    
    //ida 0 
    tk::dnn::DeformConv2d   *ida_0_p_1_dcn = new tk::dnn::DeformConv2d(net, 256, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_0_p_1_dcn_bin).c_str(), (wgs_path + ida_0_p_1_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_0_p_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       *ida_0_up_1_deconv = new tk::dnn::DeConv2d(net, 256, 4, 4, 2, 2, 1, 1, (wgs_path + ida_0_up_1_deconv_bin).c_str(), false, 256);
    tk::dnn::Shortcut       *ida_0_shortcut = new tk::dnn::Shortcut(net, base5);    
    tk::dnn::DeformConv2d   *ida_0_n_1_dcn = new tk::dnn::DeformConv2d(net, 256, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_0_n_1_dcn_bin).c_str(), (wgs_path + ida_0_n_1_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_0_n_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    ida1 = ida_0_n_1_relu;

    //ida1-1
    tk::dnn::Layer          *route_ida1_layers_1[1] = { base5 };
    tk::dnn::Route          *route_ida1_1 = new tk::dnn::Route(net, route_ida1_layers_1, 1);
    
    tk::dnn::DeformConv2d   *ida_1_p_1_dcn = new tk::dnn::DeformConv2d(net, 128, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_1_p_1_dcn_bin).c_str(), (wgs_path + ida_1_p_1_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_1_p_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       *ida_1_up_1_deconv = new tk::dnn::DeConv2d(net, 128, 4, 4, 2, 2, 1, 1, (wgs_path + ida_1_up_1_deconv_bin).c_str(), false, 128);
    tk::dnn::Shortcut       *ida_1_shortcut1 = new tk::dnn::Shortcut(net, base4);    
    tk::dnn::DeformConv2d   *ida_1_n_1_dcn = new tk::dnn::DeformConv2d(net, 128, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_1_n_1_dcn_bin).c_str(), (wgs_path + ida_1_n_1_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_1_n_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    ida2_1 = ida_1_n_1_relu;

    //ida1-2
    tk::dnn::Layer          *route_ida1_layers_2[1] = { ida1 };
    tk::dnn::Route          *route_ida1_2 = new tk::dnn::Route(net, route_ida1_layers_2, 1);

    tk::dnn::DeformConv2d   *ida_1_p_2_dcn = new tk::dnn::DeformConv2d(net, 128, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_1_p_2_dcn_bin).c_str(), (wgs_path + ida_1_p_2_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_1_p_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       *ida_1_up_2_deconv = new tk::dnn::DeConv2d(net, 128, 4, 4, 2, 2, 1, 1, (wgs_path + ida_1_up_2_deconv_bin).c_str(), false, 128);
    tk::dnn::Shortcut       *ida_1_shortcut2 = new tk::dnn::Shortcut(net, ida2_1);    
    tk::dnn::DeformConv2d   *ida_1_n_2_dcn = new tk::dnn::DeformConv2d(net, 128, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_1_n_2_dcn_bin).c_str(), (wgs_path + ida_1_n_2_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_1_n_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    ida2_2 = ida_1_n_2_relu;

    //ida2-1
    tk::dnn::Layer          *route_ida2_layers_1[1] = { base4 };
    tk::dnn::Route          *route_ida2_1 = new tk::dnn::Route(net, route_ida2_layers_1, 1);
    
    tk::dnn::DeformConv2d   *ida_2_p_1_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_p_1_dcn_bin).c_str(), (wgs_path + ida_2_p_1_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_2_p_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       *ida_2_up_1_deconv = new tk::dnn::DeConv2d(net, 64, 4, 4, 2, 2, 1, 1, (wgs_path + ida_2_up_1_deconv_bin).c_str(), false, 64);
    tk::dnn::Shortcut       *ida_2_shortcut1 = new tk::dnn::Shortcut(net, base3);    
    tk::dnn::DeformConv2d   *ida_2_n_1_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_n_1_dcn_bin).c_str(), (wgs_path + ida_2_n_1_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_2_n_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    ida3_1 = ida_2_n_1_relu;

    //ida2-2
    tk::dnn::Layer          *route_ida2_layers_2[1] = { ida2_1 };
    tk::dnn::Route          *route_ida2_2 = new tk::dnn::Route(net, route_ida2_layers_2, 1);
    
    tk::dnn::DeformConv2d   *ida_2_p_2_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_p_2_dcn_bin).c_str(), (wgs_path + ida_2_p_2_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_2_p_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       *ida_2_up_2_deconv = new tk::dnn::DeConv2d(net, 64, 4, 4, 2, 2, 1, 1, (wgs_path + ida_2_up_2_deconv_bin).c_str(), false, 64);
    tk::dnn::Shortcut       *ida_2_shortcut2 = new tk::dnn::Shortcut(net, ida3_1);    
    tk::dnn::DeformConv2d   *ida_2_n_2_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_n_2_dcn_bin).c_str(), (wgs_path + ida_2_n_2_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_2_n_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    ida3_2 = ida_2_n_2_relu;

    //ida2-3
    tk::dnn::Layer          *route_ida2_layers_3[1] = { ida2_2 };
    tk::dnn::Route          *route_ida2_3 = new tk::dnn::Route(net, route_ida2_layers_3, 1);
    
    tk::dnn::DeformConv2d   *ida_2_p_3_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_p_3_dcn_bin).c_str(), (wgs_path + ida_2_p_3_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_2_p_3_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       *ida_2_up_3_deconv = new tk::dnn::DeConv2d(net, 64, 4, 4, 2, 2, 1, 1, (wgs_path + ida_2_up_3_deconv_bin).c_str(), false, 64);
    tk::dnn::Shortcut       *ida_2_shortcut3 = new tk::dnn::Shortcut(net, ida3_2);    
    tk::dnn::DeformConv2d   *ida_2_n_3_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_n_3_dcn_bin).c_str(), (wgs_path + ida_2_n_3_conv_bin).c_str(), true);
    tk::dnn::Activation     *ida_2_n_3_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    ida3_3 = ida_2_n_3_relu;

    //idaup-1
    tk::dnn::Layer          *route_idaup_layers_1[1] = { ida2_2 };
    tk::dnn::Route          *route_idaup_1 = new tk::dnn::Route(net, route_idaup_layers_1, 1);
    
    tk::dnn::DeformConv2d   *idaup_p_1_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_up_p_1_dcn_bin).c_str(), (wgs_path + ida_up_p_1_conv_bin).c_str(), true);
    tk::dnn::Activation     *idaup_p_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       *idaup_up_1_deconv = new tk::dnn::DeConv2d(net, 64, 4, 4, 2, 2, 1, 1, (wgs_path + ida_up_up_1_deconv_bin).c_str(), false, 64);
    tk::dnn::Shortcut       *idaup_shortcut1 = new tk::dnn::Shortcut(net, ida3_3);    
    tk::dnn::DeformConv2d   *idaup_n_1_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_up_n_1_dcn_bin).c_str(), (wgs_path + ida_up_n_1_conv_bin).c_str(), true);
    tk::dnn::Activation     *idaup_n_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    idaup_1 = idaup_n_1_relu;

    //idaup-2
    tk::dnn::Layer          *route_idaup_layers_2[1] = { ida1 };
    tk::dnn::Route          *route_idaup_2 = new tk::dnn::Route(net, route_idaup_layers_2, 1);

    tk::dnn::DeformConv2d   *idaup_p_2_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_up_p_2_dcn_bin).c_str(), (wgs_path + ida_up_p_2_conv_bin).c_str(), true);
    tk::dnn::Activation     *idaup_p_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       *idaup_up_2_deconv = new tk::dnn::DeConv2d(net, 64, 8, 8, 4, 4, 2, 2, (wgs_path + ida_up_up_2_deconv_bin).c_str(), false, 64);
    tk::dnn::Shortcut       *idaup_shortcut2 = new tk::dnn::Shortcut(net, idaup_1);    
    tk::dnn::DeformConv2d   *idaup_n_2_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_up_n_2_dcn_bin).c_str(), (wgs_path + ida_up_n_2_conv_bin).c_str(), true);
    tk::dnn::Activation     *idaup_n_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    idaup_2 = idaup_n_2_relu;

    tk::dnn::Layer    *route_1_0_layers[1] = { idaup_2 };

    // hm
    tk::dnn::Conv2d     *hm_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + hm_conv1_bin).c_str(), false);
    tk::dnn::Activation *hm_relu1      = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *hm = new tk::dnn::Conv2d(net, 80, 1, 1, 1, 1, 0, 0, (wgs_path + hm_conv2_bin).c_str(), false);
    int kernel = 3; 
    int pad = (kernel - 1)/2;
    tk::dnn::Activation *hm_sig      = new tk::dnn::Activation(net, tk::dnn::ACTIVATION_LOGISTIC);
    hm_sig->setFinal();
    tk::dnn::Pooling  *hmax                 = new tk::dnn::Pooling(net, kernel, kernel, 1, 1, pad, pad, tk::dnn::POOLING_MAX);
    hmax->setFinal();

    // // wh
    tk::dnn::Route    *route_1_0             = new tk::dnn::Route(net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *wh_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + wh_conv1_bin).c_str(), false);
    tk::dnn::Activation *wh_relu1      = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *wh = new tk::dnn::Conv2d(net, 2, 1, 1, 1, 1, 0, 0, (wgs_path + wh_conv2_bin).c_str(), false);        
    wh->setFinal();
    
    // // reg
    tk::dnn::Route    *route_2_0             = new tk::dnn::Route(net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *reg_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + reg_conv1_bin).c_str(), false);
    tk::dnn::Activation *reg_relu1      = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *reg = new tk::dnn::Conv2d(net, 2, 1, 1, 1, 1, 0, 0, (wgs_path + reg_conv2_bin).c_str(), false);
    reg->setFinal();
#pragma GCC diagnostic pop
	input_dim.width = net->input_dim.w;
	input_dim.height = net->input_dim.h;
	input_dim.channel = net->input_dim.c;

	net->fileImgList = centernetAppConfig.calib_image_path;
	net->num_calib_images = centernetAppConfig.calib_images_num;

	jedi_network->net = net;

	return jedi_network;
}


//tk::dnn::Network *CenternetApplication::createNetwork(ConfigInstance *basic_config_data)
//{
//	std::string name_path = centernetAppConfig.name_path;
//	std::string bin_path(basic_config_data->bin_path);
//    std::string wgs_path  = bin_path + "/layers/";
//
//	tk::dnn::dataDim_t dim(1, 3, 512, 512, 1);
//    tk::dnn::Network *net = new tk::dnn::Network(dim);
//    tk::dnn::Layer *last1, *last2, *last3, *last4;
//    tk::dnn::Layer *base2, *base3, *base4, *base5, *ida1, *ida2_1, *ida2_2, *ida3_1, *ida3_2, *ida3_3, *idaup_1, *idaup_2;
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wunused-variable"
//    tk::dnn::Conv2d *conv1 = new tk::dnn::Conv2d(net, 16, 7, 7, 1, 1, 3, 3, (wgs_path + conv1_bin).c_str(), true);
//    tk::dnn::Activation *relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//
//    tk::dnn::Conv2d *conv2 = new tk::dnn::Conv2d(net, 16, 3, 3, 1, 1, 1, 1, (wgs_path + conv2_bin).c_str(), true);
//    tk::dnn::Activation *relu2 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    //base1 = relu2;
//
//    tk::dnn::Conv2d *conv3 = new tk::dnn::Conv2d(net, 32, 3, 3, 2, 2, 1, 1, (wgs_path + conv3_bin).c_str(), true);
//    tk::dnn::Activation *relu3 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    base2 = relu3;
//    
//    // level 2
//    // tree 1
//        tk::dnn::Conv2d     *s1_t1_conv1 = new tk::dnn::Conv2d(net, 64, 3, 3, 2, 2, 1, 1, (wgs_path + s1_t1_conv1_bin).c_str(), true);
//        tk::dnn::Activation *s1_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//       
//        tk::dnn::Conv2d     *s1_t1_conv2 = new tk::dnn::Conv2d(net, 64, 3, 3, 1, 1, 1, 1, (wgs_path + s1_t1_conv2_bin).c_str(), true);
//        last2 = s1_t1_conv2;
//
//        // get the basicblock input and apply maxpool conv2d and relu
//        tk::dnn::Layer      *route_s1_t1_layers[1] = { base2 };
//        tk::dnn::Route      *route_s1_t1 = new tk::dnn::Route(net, route_s1_t1_layers, 1);
//        // downsample
//        tk::dnn::Pooling   *s1_t1_maxpool = new tk::dnn::Pooling(net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
//        // project
//        tk::dnn::Conv2d    *s1_t1_residual1_conv1 = new tk::dnn::Conv2d(net, 64, 1, 1, 1, 1, 0, 0, (wgs_path + s1_t1_project).c_str(), true);
//        
//        tk::dnn::Shortcut  *s1_t1_s1 = new tk::dnn::Shortcut(net, last2);
//        tk::dnn::Activation *s1_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    
//    last1 = s1_t1_relu;
//    // tree 2
//        tk::dnn::Conv2d     *s1_t2_conv1 = new tk::dnn::Conv2d(net, 64, 3, 3, 1, 1, 1, 1, (wgs_path + s1_t2_conv1_bin).c_str(), true);
//        tk::dnn::Activation *s1_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//       
//        tk::dnn::Conv2d     *s1_t2_conv2 = new tk::dnn::Conv2d(net, 64, 3, 3, 1, 1, 1, 1, (wgs_path + s1_t2_conv2_bin).c_str(), true);
//        
//        tk::dnn::Shortcut   *s1_t2_s1 = new tk::dnn::Shortcut(net, last1);
//        tk::dnn::Activation *s1_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//        last2 = s1_t2_relu;
//
//    // root
//        // join last1 and net in single input 128, 56, 56
//        tk::dnn::Layer      *route_s1_root_layers[2] = { last2, last1 };
//        tk::dnn::Route      *route_s1_root = new tk::dnn::Route(net, route_s1_root_layers, 2);
//        tk::dnn::Conv2d     *s1_root_conv1 = new tk::dnn::Conv2d(net, 64, 1, 1, 1, 1, 0, 0, (wgs_path + s1_root_conv1_bin).c_str(), true);
//        tk::dnn::Activation *s1_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//
//    base3 = s1_root_relu;
//
//    // level 3
//    // tree 1
//        // tree 1
//            tk::dnn::Conv2d     *s2_t1_t1_conv1 = new tk::dnn::Conv2d(net, 128, 3, 3, 2, 2, 1, 1, (wgs_path + s2_t1_t1_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s2_t1_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);      
//        
//            tk::dnn::Conv2d     *s2_t1_t1_conv2 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t1_t1_conv2_bin).c_str(), true);
//            last2 = s2_t1_t1_conv2;
//
//            // get the basicblock input and apply maxpool conv2d and relu
//            tk::dnn::Layer      *route_s2_t1_t1_layers[1] = { base3 };
//            tk::dnn::Route      *route_s2_t1_t1 = new tk::dnn::Route(net, route_s2_t1_t1_layers, 1);
//            // downsample
//            tk::dnn::Pooling    *s2_t1_t1_maxpool1 = new tk::dnn::Pooling(net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
//            last4 = s2_t1_t1_maxpool1;
//            // project
//            tk::dnn::Conv2d     *s2_t1_t1_residual1_conv1 = new tk::dnn::Conv2d(net, 128, 1, 1, 1, 1, 0, 0, (wgs_path + s2_t1_t1_project).c_str(), true);
//            
//            tk::dnn::Shortcut   *s2_t1_t1_s1 = new tk::dnn::Shortcut(net, last2);
//            tk::dnn::Activation *s2_t1_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//
//        last1 = s2_t1_t1_relu;
//        
//        // tree 2
//            tk::dnn::Conv2d     *s2_t1_t2_conv1 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t1_t2_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s2_t1_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//        
//            tk::dnn::Conv2d     *s2_t1_t2_conv2 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t1_t2_conv2_bin).c_str(), true);
//            
//            tk::dnn::Shortcut   *s2_t1_t2_s1 = new tk::dnn::Shortcut(net, last1);
//            tk::dnn::Activation *s2_t1_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//            last2 = s2_t1_t2_relu;
//
//        // root
//            // join last1 and net in single input 128, 56, 56
//            tk::dnn::Layer      *route_s2_t1_root_layers[2] = { last2, last1 };
//            tk::dnn::Route      *route_s2_t1_root = new tk::dnn::Route(net, route_s2_t1_root_layers, 2);
//            tk::dnn::Conv2d     *s2_t1_root_conv1 = new tk::dnn::Conv2d(net, 128, 1, 1, 1, 1, 0, 0, (wgs_path + s2_t1_root_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s2_t1_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    
//    last1 = s2_t1_root_relu;
//    last3 = s2_t1_root_relu;
//    // tree 2
//        // tree 1
//            tk::dnn::Conv2d     *s2_t2_t1_conv1 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t2_t1_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s2_t2_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//        
//            tk::dnn::Conv2d     *s2_t2_t1_conv2 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t2_t1_conv2_bin).c_str(), true);
//            tk::dnn::Shortcut   *s2_t2_t1_s1 = new tk::dnn::Shortcut(net, last1);
//            tk::dnn::Activation *s2_t2_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//
//        last1 = s2_t2_t1_relu;
//        
//        // tree 2
//            tk::dnn::Conv2d     *s2_t2_t2_conv1 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t2_t2_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s2_t2_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//        
//            tk::dnn::Conv2d     *s2_t2_t2_conv2 = new tk::dnn::Conv2d(net, 128, 3, 3, 1, 1, 1, 1, (wgs_path + s2_t2_t2_conv2_bin).c_str(), true);
//            
//            tk::dnn::Shortcut   *s2_t2_t2_s1 = new tk::dnn::Shortcut(net, last1);
//            tk::dnn::Activation *s2_t2_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//            last2 = s2_t2_t2_relu;
//
//        // root
//            // join last1 and net in single input 128, 56, 56
//            tk::dnn::Layer      *route_s2_t2_root_layers[4] = { last2, last1, last4, last3};
//            tk::dnn::Route      *route_s2_t2_root = new tk::dnn::Route(net, route_s2_t2_root_layers, 4);
//            tk::dnn::Conv2d     *s2_t2_root_conv1 = new tk::dnn::Conv2d(net, 128, 1, 1, 1, 1, 0, 0, (wgs_path + s2_t2_root_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s2_t2_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    
//    base4 = s2_t2_root_relu;
//
//    // level 4
//    // tree 1
//        // tree 1
//            tk::dnn::Conv2d     *s3_t1_t1_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 2, 2, 1, 1, (wgs_path + s3_t1_t1_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s3_t1_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);      
//        
//            tk::dnn::Conv2d     *s3_t1_t1_conv2 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t1_t1_conv2_bin).c_str(), true);
//            last2 = s3_t1_t1_conv2;
//
//            // get the basicblock input and apply maxpool conv2d and relu
//            tk::dnn::Layer      *route_s3_t1_t1_layers[1] = { base4 };
//            tk::dnn::Route      *route_s3_t1_t1 = new tk::dnn::Route(net, route_s3_t1_t1_layers, 1);
//            // downsample
//            tk::dnn::Pooling    *s3_t1_t1_maxpool1 = new tk::dnn::Pooling(net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
//            last4 = s3_t1_t1_maxpool1;
//            // project
//            tk::dnn::Conv2d     *s3_t1_t1_residual1_conv1 = new tk::dnn::Conv2d(net, 256, 1, 1, 1, 1, 0, 0, (wgs_path + s3_t1_t1_project).c_str(), true);
//            
//            tk::dnn::Shortcut   *s3_t1_t1_s1 = new tk::dnn::Shortcut(net, last2);
//            tk::dnn::Activation *s3_t1_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//
//        last1 = s3_t1_t1_relu;
//        
//        // tree 2
//            tk::dnn::Conv2d     *s3_t1_t2_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t1_t2_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s3_t1_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//        
//            tk::dnn::Conv2d     *s3_t1_t2_conv2 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t1_t2_conv2_bin).c_str(), true);
//            
//            tk::dnn::Shortcut   *s3_t1_t2_s1 = new tk::dnn::Shortcut(net, last1);
//            tk::dnn::Activation *s3_t1_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//            last2 = s3_t1_t2_relu;
//
//        // root
//            // join last1 and net in single input 256, 56, 56
//            tk::dnn::Layer      *route_s3_t1_root_layers[2] = { last2, last1 };
//            tk::dnn::Route      *route_s3_t1_root = new tk::dnn::Route(net, route_s3_t1_root_layers, 2);
//            tk::dnn::Conv2d     *s3_t1_root_conv1 = new tk::dnn::Conv2d(net, 256, 1, 1, 1, 1, 0, 0, (wgs_path + s3_t1_root_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s3_t1_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    
//    last1 = s3_t1_root_relu;
//    last3 = s3_t1_root_relu;
//    // tree 2
//        // tree 1
//            tk::dnn::Conv2d     *s3_t2_t1_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t2_t1_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s3_t2_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//        
//            tk::dnn::Conv2d     *s3_t2_t1_conv2 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t2_t1_conv2_bin).c_str(), true);
//            tk::dnn::Shortcut   *s3_t2_t1_s1 = new tk::dnn::Shortcut(net, last1);
//            tk::dnn::Activation *s3_t2_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//
//        last1 = s3_t2_t1_relu;
//        
//        // tree 2
//            tk::dnn::Conv2d     *s3_t2_t2_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t2_t2_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s3_t2_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//        
//            tk::dnn::Conv2d     *s3_t2_t2_conv2 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + s3_t2_t2_conv2_bin).c_str(), true);
//            
//            tk::dnn::Shortcut   *s3_t2_t2_s1 = new tk::dnn::Shortcut(net, last1);
//            tk::dnn::Activation *s3_t2_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//            last2 = s3_t2_t2_relu;
//
//        // root
//            // join last1 and net in single input 256, 56, 56
//            tk::dnn::Layer      *route_s3_t2_root_layers[4] = { last2, last1, last4, last3};
//            tk::dnn::Route      *route_s3_t2_root = new tk::dnn::Route(net, route_s3_t2_root_layers, 4);
//            tk::dnn::Conv2d     *s3_t2_root_conv1 = new tk::dnn::Conv2d(net, 256, 1, 1, 1, 1, 0, 0, (wgs_path + s3_t2_root_conv1_bin).c_str(), true);
//            tk::dnn::Activation *s3_t2_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    
//    base5 = s3_t2_root_relu;
//
//    // level 5
//    // tree 1
//        tk::dnn::Conv2d     *s4_t1_conv1 = new tk::dnn::Conv2d(net, 512, 3, 3, 2, 2, 1, 1, (wgs_path + s4_t1_conv1_bin).c_str(), true);
//        tk::dnn::Activation *s4_t1_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//       
//        tk::dnn::Conv2d     *s4_t1_conv2 = new tk::dnn::Conv2d(net, 512, 3, 3, 1, 1, 1, 1, (wgs_path + s4_t1_conv2_bin).c_str(), true);
//        last2 = s4_t1_conv2;
//
//        // get the basicblock input and apply maxpool conv2d and relu
//        tk::dnn::Layer      *route_s4_t1_layers[1] = { base5 };
//        tk::dnn::Route      *route_s4_t1 = new tk::dnn::Route(net, route_s4_t1_layers, 1);
//        // downsample
//        tk::dnn::Pooling    *s4_t1_maxpool1 = new tk::dnn::Pooling(net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
//        last4 = s4_t1_maxpool1;
//        // project
//        tk::dnn::Conv2d     *s4_t1_residual1_conv1 = new tk::dnn::Conv2d(net, 512, 1, 1, 1, 1, 0, 0, (wgs_path + s4_t1_project).c_str(), true);
//        
//        tk::dnn::Shortcut   *s4_t1_s1 = new tk::dnn::Shortcut(net, last2);
//        tk::dnn::Activation *s4_t1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//
//    last1 = s4_t1_relu;
//    
//    // tree 2
//        tk::dnn::Conv2d     *s4_t2_conv1 = new tk::dnn::Conv2d(net, 512, 3, 3, 1, 1, 1, 1, (wgs_path + s4_t2_conv1_bin).c_str(), true);
//        tk::dnn::Activation *s4_t2_relu1 = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);       
//       
//        tk::dnn::Conv2d     *s4_t2_conv2 = new tk::dnn::Conv2d(net, 512, 3, 3, 1, 1, 1, 1, (wgs_path + s4_t2_conv2_bin).c_str(), true);
//        
//        tk::dnn::Shortcut   *s4_t2_s1 = new tk::dnn::Shortcut(net, last1);
//        tk::dnn::Activation *s4_t2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//        last2 = s4_t2_relu;
//
//    // root
//        // join last1 and net in single input 128, 56, 56
//        tk::dnn::Layer      *route_s4_root_layers[3] = { last2, last1, last4 };
//        tk::dnn::Route      *route_s4_root = new tk::dnn::Route(net, route_s4_root_layers, 3);
//        tk::dnn::Conv2d     *s4_root_conv1 = new tk::dnn::Conv2d(net, 512, 1, 1, 1, 1, 0, 0, (wgs_path + s4_root_conv1_bin).c_str(), true);
//        tk::dnn::Activation *s4_root_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//
//    //base6 = s4_root_relu;
//    
//    //final
//    // tk::dnn::Pooling avgpool(net, 7, 7, 7, 7, 0, 0, tk::dnn::POOLING_AVERAGE);
//    // tk::dnn::Dense   fc(net, 1000, fc_bin);
//    
//    //ida 0 
//    tk::dnn::DeformConv2d   *ida_0_p_1_dcn = new tk::dnn::DeformConv2d(net, 256, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_0_p_1_dcn_bin).c_str(), (wgs_path + ida_0_p_1_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_0_p_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::DeConv2d       *ida_0_up_1_deconv = new tk::dnn::DeConv2d(net, 256, 4, 4, 2, 2, 1, 1, (wgs_path + ida_0_up_1_deconv_bin).c_str(), false, 256);
//    tk::dnn::Shortcut       *ida_0_shortcut = new tk::dnn::Shortcut(net, base5);    
//    tk::dnn::DeformConv2d   *ida_0_n_1_dcn = new tk::dnn::DeformConv2d(net, 256, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_0_n_1_dcn_bin).c_str(), (wgs_path + ida_0_n_1_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_0_n_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    ida1 = ida_0_n_1_relu;
//
//    //ida1-1
//    tk::dnn::Layer          *route_ida1_layers_1[1] = { base5 };
//    tk::dnn::Route          *route_ida1_1 = new tk::dnn::Route(net, route_ida1_layers_1, 1);
//    
//    tk::dnn::DeformConv2d   *ida_1_p_1_dcn = new tk::dnn::DeformConv2d(net, 128, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_1_p_1_dcn_bin).c_str(), (wgs_path + ida_1_p_1_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_1_p_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::DeConv2d       *ida_1_up_1_deconv = new tk::dnn::DeConv2d(net, 128, 4, 4, 2, 2, 1, 1, (wgs_path + ida_1_up_1_deconv_bin).c_str(), false, 128);
//    tk::dnn::Shortcut       *ida_1_shortcut1 = new tk::dnn::Shortcut(net, base4);    
//    tk::dnn::DeformConv2d   *ida_1_n_1_dcn = new tk::dnn::DeformConv2d(net, 128, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_1_n_1_dcn_bin).c_str(), (wgs_path + ida_1_n_1_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_1_n_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    ida2_1 = ida_1_n_1_relu;
//
//    //ida1-2
//    tk::dnn::Layer          *route_ida1_layers_2[1] = { ida1 };
//    tk::dnn::Route          *route_ida1_2 = new tk::dnn::Route(net, route_ida1_layers_2, 1);
//
//    tk::dnn::DeformConv2d   *ida_1_p_2_dcn = new tk::dnn::DeformConv2d(net, 128, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_1_p_2_dcn_bin).c_str(), (wgs_path + ida_1_p_2_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_1_p_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::DeConv2d       *ida_1_up_2_deconv = new tk::dnn::DeConv2d(net, 128, 4, 4, 2, 2, 1, 1, (wgs_path + ida_1_up_2_deconv_bin).c_str(), false, 128);
//    tk::dnn::Shortcut       *ida_1_shortcut2 = new tk::dnn::Shortcut(net, ida2_1);    
//    tk::dnn::DeformConv2d   *ida_1_n_2_dcn = new tk::dnn::DeformConv2d(net, 128, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_1_n_2_dcn_bin).c_str(), (wgs_path + ida_1_n_2_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_1_n_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    ida2_2 = ida_1_n_2_relu;
//
//    //ida2-1
//    tk::dnn::Layer          *route_ida2_layers_1[1] = { base4 };
//    tk::dnn::Route          *route_ida2_1 = new tk::dnn::Route(net, route_ida2_layers_1, 1);
//    
//    tk::dnn::DeformConv2d   *ida_2_p_1_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_p_1_dcn_bin).c_str(), (wgs_path + ida_2_p_1_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_2_p_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::DeConv2d       *ida_2_up_1_deconv = new tk::dnn::DeConv2d(net, 64, 4, 4, 2, 2, 1, 1, (wgs_path + ida_2_up_1_deconv_bin).c_str(), false, 64);
//    tk::dnn::Shortcut       *ida_2_shortcut1 = new tk::dnn::Shortcut(net, base3);    
//    tk::dnn::DeformConv2d   *ida_2_n_1_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_n_1_dcn_bin).c_str(), (wgs_path + ida_2_n_1_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_2_n_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    ida3_1 = ida_2_n_1_relu;
//
//    //ida2-2
//    tk::dnn::Layer          *route_ida2_layers_2[1] = { ida2_1 };
//    tk::dnn::Route          *route_ida2_2 = new tk::dnn::Route(net, route_ida2_layers_2, 1);
//    
//    tk::dnn::DeformConv2d   *ida_2_p_2_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_p_2_dcn_bin).c_str(), (wgs_path + ida_2_p_2_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_2_p_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::DeConv2d       *ida_2_up_2_deconv = new tk::dnn::DeConv2d(net, 64, 4, 4, 2, 2, 1, 1, (wgs_path + ida_2_up_2_deconv_bin).c_str(), false, 64);
//    tk::dnn::Shortcut       *ida_2_shortcut2 = new tk::dnn::Shortcut(net, ida3_1);    
//    tk::dnn::DeformConv2d   *ida_2_n_2_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_n_2_dcn_bin).c_str(), (wgs_path + ida_2_n_2_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_2_n_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    ida3_2 = ida_2_n_2_relu;
//
//    //ida2-3
//    tk::dnn::Layer          *route_ida2_layers_3[1] = { ida2_2 };
//    tk::dnn::Route          *route_ida2_3 = new tk::dnn::Route(net, route_ida2_layers_3, 1);
//    
//    tk::dnn::DeformConv2d   *ida_2_p_3_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_p_3_dcn_bin).c_str(), (wgs_path + ida_2_p_3_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_2_p_3_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::DeConv2d       *ida_2_up_3_deconv = new tk::dnn::DeConv2d(net, 64, 4, 4, 2, 2, 1, 1, (wgs_path + ida_2_up_3_deconv_bin).c_str(), false, 64);
//    tk::dnn::Shortcut       *ida_2_shortcut3 = new tk::dnn::Shortcut(net, ida3_2);    
//    tk::dnn::DeformConv2d   *ida_2_n_3_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_2_n_3_dcn_bin).c_str(), (wgs_path + ida_2_n_3_conv_bin).c_str(), true);
//    tk::dnn::Activation     *ida_2_n_3_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    ida3_3 = ida_2_n_3_relu;
//
//    //idaup-1
//    tk::dnn::Layer          *route_idaup_layers_1[1] = { ida2_2 };
//    tk::dnn::Route          *route_idaup_1 = new tk::dnn::Route(net, route_idaup_layers_1, 1);
//    
//    tk::dnn::DeformConv2d   *idaup_p_1_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_up_p_1_dcn_bin).c_str(), (wgs_path + ida_up_p_1_conv_bin).c_str(), true);
//    tk::dnn::Activation     *idaup_p_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::DeConv2d       *idaup_up_1_deconv = new tk::dnn::DeConv2d(net, 64, 4, 4, 2, 2, 1, 1, (wgs_path + ida_up_up_1_deconv_bin).c_str(), false, 64);
//    tk::dnn::Shortcut       *idaup_shortcut1 = new tk::dnn::Shortcut(net, ida3_3);    
//    tk::dnn::DeformConv2d   *idaup_n_1_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_up_n_1_dcn_bin).c_str(), (wgs_path + ida_up_n_1_conv_bin).c_str(), true);
//    tk::dnn::Activation     *idaup_n_1_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    idaup_1 = idaup_n_1_relu;
//
//    //idaup-2
//    tk::dnn::Layer          *route_idaup_layers_2[1] = { ida1 };
//    tk::dnn::Route          *route_idaup_2 = new tk::dnn::Route(net, route_idaup_layers_2, 1);
//
//    tk::dnn::DeformConv2d   *idaup_p_2_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_up_p_2_dcn_bin).c_str(), (wgs_path + ida_up_p_2_conv_bin).c_str(), true);
//    tk::dnn::Activation     *idaup_p_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::DeConv2d       *idaup_up_2_deconv = new tk::dnn::DeConv2d(net, 64, 8, 8, 4, 4, 2, 2, (wgs_path + ida_up_up_2_deconv_bin).c_str(), false, 64);
//    tk::dnn::Shortcut       *idaup_shortcut2 = new tk::dnn::Shortcut(net, idaup_1);    
//    tk::dnn::DeformConv2d   *idaup_n_2_dcn = new tk::dnn::DeformConv2d(net, 64, 1, 3, 3, 1, 1, 1, 1, (wgs_path + ida_up_n_2_dcn_bin).c_str(), (wgs_path + ida_up_n_2_conv_bin).c_str(), true);
//    tk::dnn::Activation     *idaup_n_2_relu = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    idaup_2 = idaup_n_2_relu;
//
//    tk::dnn::Layer    *route_1_0_layers[1] = { idaup_2 };
//
//    // hm
//    tk::dnn::Conv2d     *hm_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + hm_conv1_bin).c_str(), false);
//    tk::dnn::Activation *hm_relu1      = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::Conv2d     *hm = new tk::dnn::Conv2d(net, 80, 1, 1, 1, 1, 0, 0, (wgs_path + hm_conv2_bin).c_str(), false);
//    int kernel = 3; 
//    int pad = (kernel - 1)/2;
//    tk::dnn::Activation *hm_sig      = new tk::dnn::Activation(net, tk::dnn::ACTIVATION_LOGISTIC);
//    hm_sig->setFinal();
//    tk::dnn::Pooling  *hmax                 = new tk::dnn::Pooling(net, kernel, kernel, 1, 1, pad, pad, tk::dnn::POOLING_MAX);
//    hmax->setFinal();
//
//    // // wh
//    tk::dnn::Route    *route_1_0             = new tk::dnn::Route(net, route_1_0_layers, 1);
//    tk::dnn::Conv2d     *wh_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + wh_conv1_bin).c_str(), false);
//    tk::dnn::Activation *wh_relu1      = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::Conv2d     *wh = new tk::dnn::Conv2d(net, 2, 1, 1, 1, 1, 0, 0, (wgs_path + wh_conv2_bin).c_str(), false);        
//    wh->setFinal();
//    
//    // // reg
//    tk::dnn::Route    *route_2_0             = new tk::dnn::Route(net, route_1_0_layers, 1);
//    tk::dnn::Conv2d     *reg_conv1 = new tk::dnn::Conv2d(net, 256, 3, 3, 1, 1, 1, 1, (wgs_path + reg_conv1_bin).c_str(), false);
//    tk::dnn::Activation *reg_relu1      = new tk::dnn::Activation(net, CUDNN_ACTIVATION_RELU);
//    tk::dnn::Conv2d     *reg = new tk::dnn::Conv2d(net, 2, 1, 1, 1, 1, 0, 0, (wgs_path + reg_conv2_bin).c_str(), false);
//    reg->setFinal();
//#pragma GCC diagnostic pop
//	input_dim.width = net->input_dim.w;
//	input_dim.height = net->input_dim.h;
//	input_dim.channel = net->input_dim.c;
//
//	net->fileImgList = centernetAppConfig.calib_image_path;
//	net->num_calib_images = centernetAppConfig.calib_images_num;
//
//	return net;
//}

void CenternetApplication::initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	this->network_name = network_name;
	dataset = new ImageDataset(centernetAppConfig.image_path);

	dst.at<float>(0,0)= input_dim.width * 0.5;
	dst.at<float>(0,1)= input_dim.height * 0.5;
	dst.at<float>(1,0)= input_dim.width * 0.5;
	dst.at<float>(1,1)= input_dim.height * 0.5 +  input_dim.width * -0.5;
	dst.at<float>(2,0)=dst.at<float>(1,0) + (-dst.at<float>(0,1)+dst.at<float>(1,1) );
	dst.at<float>(2,1)=dst.at<float>(1,1) + (dst.at<float>(0,0)-dst.at<float>(1,0) );
}

void CenternetApplication::preprocessing(int thread_id, int sample_index, int batch_index, IN OUT float *input_buffer)
{
	int image_index = (sample_index + batch_index) % dataset->getSize();

	ImageData *image_data = dataset->getData(image_index);
	int orignal_width = 0;
	int original_height = 0;

	loadImageAffineTransform((char *)(image_data->path.c_str()), input_dim.width, input_dim.height, input_dim.channel,
			&orignal_width, &original_height, mean, stddev, dst, input_buffer);

	image_data->width = orignal_width;
	image_data->height = original_height;

}


void CenternetApplication::printBox(int sample_index, int batch, Detection *dets, std::vector<int> detections_num)
{
	for(int iter1 = 0; iter1 < batch; iter1++) {
		int image_index = (sample_index * batch + iter1) % dataset->getSize();
		ImageData *data = dataset->getData(image_index);
		char *path = (char *)(data->path.c_str());

		detectCOCO(&dets[iter1 * NBOXES], detections_num[iter1], image_index, data->width, data->height, input_dim.width, input_dim.height, path);
	}
}


void CenternetApplication::initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number)
{
	int width = 128;
	dst2.at<float>(0,0)=width * 0.5;
	dst2.at<float>(0,1)=width * 0.5;
	dst2.at<float>(1,0)=width * 0.5;
	dst2.at<float>(1,1)=width * 0.5 +  width * -0.5;

	dst2.at<float>(2,0)=dst2.at<float>(1,0) + (-dst2.at<float>(0,1)+dst2.at<float>(1,1) );
	dst2.at<float>(2,1)=dst2.at<float>(1,1) + (dst2.at<float>(0,0)-dst2.at<float>(1,0) );
    ids_d = cuda_make_int_array(NULL, dim_hm.c * dim_hm.h * dim_hm.w);

	for(int i = 0 ; i < thread_number ; i++) {
		cudaStream_t stream;
		check_error(cudaStreamCreate(&stream));
		post_streams.push_back(stream);

		Detection *dets;
		allocateDetectionBox(maximum_batch_size, &dets);
		dets_vec.push_back(dets);

		this->detection_num_vec.push_back(std::vector<int>(maximum_batch_size, 0));

		CenterPostProcessingSharedData *cpuData =(CenterPostProcessingSharedData *) malloc(sizeof(CenterPostProcessingSharedData));
		CenterPostProcessingGPUData *gpuData = (CenterPostProcessingGPUData *) malloc(sizeof(CenterPostProcessingGPUData));

		cpuData->scores = cuda_make_array_host(K_VALUE);
		checkCuda( cudaHostGetDevicePointer(&(gpuData->shared.scores), cpuData->scores, 0));

		cpuData->clses = cuda_make_int_array_host(K_VALUE);
		checkCuda( cudaHostGetDevicePointer(&(gpuData->shared.clses), cpuData->clses, 0));

		gpuData->topk_inds = cuda_make_int_array(NULL, K_VALUE);
		gpuData->topk_ys = cuda_make_array(NULL, K_VALUE);
		gpuData->topk_xs = cuda_make_array(NULL, K_VALUE);
		gpuData->inttopk_ys = cuda_make_int_array(NULL, K_VALUE);
		gpuData->inttopk_xs = cuda_make_int_array(NULL, K_VALUE);
		gpuData->src_out = cuda_make_array(NULL, K_VALUE);
		gpuData->ids_out = cuda_make_int_array(NULL, K_VALUE);

		cpuData->bbx0 = cuda_make_array_host(K_VALUE);
		checkCuda( cudaHostGetDevicePointer(&(gpuData->shared.bbx0), cpuData->bbx0, 0));
		cpuData->bby0 = cuda_make_array_host(K_VALUE);
		checkCuda( cudaHostGetDevicePointer(&(gpuData->shared.bby0), cpuData->bby0, 0));
		cpuData->bbx1 = cuda_make_array_host(K_VALUE);
		checkCuda( cudaHostGetDevicePointer(&(gpuData->shared.bbx1), cpuData->bbx1, 0));
		cpuData->bby1 = cuda_make_array_host(K_VALUE);
		checkCuda( cudaHostGetDevicePointer(&(gpuData->shared.bby1), cpuData->bby1, 0));

		cpuDataList.push_back(cpuData);
		gpuDataList.push_back(gpuData);

	    float *target_coords = (float *) malloc(4 * K_VALUE * sizeof(float));
	    target_coords_vec.push_back(target_coords);

	}
}

void CenternetApplication::postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch)
{
	float **rt_batch_out = (float **) malloc(sizeof(float *) * output_num);
	//float **rt_out = (float **) malloc(sizeof(float *) * output_num);
	cudaStream_t stream = post_streams[thread_id];
	CenterPostProcessingSharedData *cpuData = cpuDataList[thread_id];
	CenterPostProcessingGPUData *gpuData = gpuDataList[thread_id];

	for(int i = 0 ; i < output_num; i++) {
		cudaHostGetDevicePointer(&(rt_batch_out[i]), output_buffers[i], 0);
	}

	for (int iter = 0 ; iter < batch ; iter++) {
		float *rt_out[4];
		rt_out[0] = rt_batch_out[0] + dim_hm.tot() * iter;
		rt_out[1] = rt_batch_out[1] + dim_hm.tot() * iter;
		rt_out[2] = rt_batch_out[2] + dim_wh.tot() * iter;
		rt_out[3] = rt_batch_out[3] + dim_reg.tot() * iter;

		checkCuda( cudaMemcpyAsync(ids_d, ids, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int), cudaMemcpyHostToDevice, stream) );

		subtractWithThreshold_cnet(stream, rt_out[0], rt_out[0] + dim_hm.tot(), rt_out[1], rt_out[0]);

		sort_cnet(stream, rt_out[0],rt_out[0]+dim_hm.tot(),ids_d);

		topk_cnet(stream, rt_out[0], ids_d, K_VALUE, gpuData->shared.scores, gpuData->topk_inds);

		topKxyclasses_cnet(stream, gpuData->topk_inds, gpuData->topk_inds+K_VALUE, K_VALUE, 128, dim_hm.w*dim_hm.h, gpuData->shared.clses, gpuData->inttopk_xs, gpuData->inttopk_ys);

		checkCuda( cudaMemcpyAsync(gpuData->topk_xs, (float *)gpuData->inttopk_xs, K_VALUE*sizeof(float), cudaMemcpyDeviceToDevice, stream) );
		checkCuda( cudaMemcpyAsync(gpuData->topk_ys, (float *)gpuData->inttopk_ys, K_VALUE*sizeof(float), cudaMemcpyDeviceToDevice, stream) );

		topKxyAddOffset_cnet(stream, gpuData->topk_inds, K_VALUE, dim_reg.h*dim_reg.w, gpuData->inttopk_xs, gpuData->inttopk_ys, gpuData->topk_xs, gpuData->topk_ys, rt_out[3], gpuData->src_out, gpuData->ids_out);

		bboxes_cnet(stream, gpuData->topk_inds, K_VALUE, dim_wh.h*dim_wh.w, gpuData->topk_xs, gpuData->topk_ys, rt_out[2], gpuData->shared.bbx0, gpuData->shared.bbx1, gpuData->shared.bby0, gpuData->shared.bby1, gpuData->src_out, gpuData->ids_out);
		checkCuda( cudaStreamSynchronize(stream) );

	    ImageData *data = dataset->getData(sample_index * batch + iter);
		int orig_width = data->width;
		int orig_height = data->height;

	    cv::Mat new_pt1(cv::Size(1,2), CV_32F);
	    cv::Mat new_pt2(cv::Size(1,2), CV_32F);

	    cv::Mat trans2 = restoreAffinedTransform(orig_width, orig_height, dst2);

	    for(int i = 0; i<K_VALUE; i++){
	        new_pt1.at<float>(0,0)=static_cast<float>(trans2.at<double>(0,0))*cpuData->bbx0[i] +
	                                static_cast<float>(trans2.at<double>(0,1))*cpuData->bby0[i] +
	                                static_cast<float>(trans2.at<double>(0,2))*1.0;
	        new_pt1.at<float>(0,1)=static_cast<float>(trans2.at<double>(1,0))*cpuData->bbx0[i] +
	                                static_cast<float>(trans2.at<double>(1,1))*cpuData->bby0[i] +
	                                static_cast<float>(trans2.at<double>(1,2))*1.0;

	        new_pt2.at<float>(0,0)=static_cast<float>(trans2.at<double>(0,0))*cpuData->bbx1[i] +
	                                static_cast<float>(trans2.at<double>(0,1))*cpuData->bby1[i] +
	                                static_cast<float>(trans2.at<double>(0,2))*1.0;
	        new_pt2.at<float>(0,1)=static_cast<float>(trans2.at<double>(1,0))*cpuData->bbx1[i] +
	                                static_cast<float>(trans2.at<double>(1,1))*cpuData->bby1[i] +
	                                static_cast<float>(trans2.at<double>(1,2))*1.0;

	        target_coords_vec[thread_id][i*4] = new_pt1.at<float>(0,0);
	        target_coords_vec[thread_id][i*4+1] = new_pt1.at<float>(0,1);
	        target_coords_vec[thread_id][i*4+2] = new_pt2.at<float>(0,0);
	        target_coords_vec[thread_id][i*4+3] = new_pt2.at<float>(0,1);
		}

		int count = 0;
        for(int i = 0; i<NUM_CLASSES; i++){
            for(int j=0; j<K_VALUE; j++) {
                if(cpuData->clses[j] == i){
                    if(cpuData->scores[j] > CONFIDENCE_THRESH){
                        float x0   = target_coords_vec[thread_id][j*4];
                        float y0   = target_coords_vec[thread_id][j*4+1];
                        float x1   = target_coords_vec[thread_id][j*4+2];
                        float y1   = target_coords_vec[thread_id][j*4+3];
                        int obj_class = cpuData->clses[j];
                        float prob = cpuData->scores[j];

                        Detection *dets = dets_vec[thread_id];
                        Detection *det = &(dets[iter * NBOXES + count]);
                        det->classes = obj_class;
						det->objectness = cpuData->scores[j];	
						memset(det->prob, 0, NUM_CLASSES * sizeof(float));
						det->prob[obj_class] = prob;

                        det->bbox.x = x0 + (x1 - x0)/2.;
                        det->bbox.y = y0 + (y1 - y0)/2.;
                        det->bbox.w = x1 - x0;
                        det->bbox.h = y1 - y0;

                        count++;
                    }
                }
            }
        }
        detection_num_vec[thread_id][iter] = count;
	}

	free(rt_batch_out);
	//free(rt_out);
}

void CenternetApplication::postprocessing2(int thread_id, int sample_index, int batch) {
	printBox(sample_index, batch, dets_vec[thread_id], detection_num_vec[thread_id]);
}

CenternetApplication::~CenternetApplication()
{
	while(post_streams.size() > 0) {
		cudaStream_t stream = post_streams.back();
		cudaStreamDestroy(stream);
		post_streams.pop_back();
	}

	while(gpuDataList.size() > 0) {
		CenterPostProcessingGPUData *gpuData = gpuDataList.back();
		cudaFree(gpuData->topk_inds);
		cudaFree(gpuData->topk_ys);
		cudaFree(gpuData->topk_xs);
		cudaFree(gpuData->inttopk_ys);
		cudaFree(gpuData->inttopk_xs);
		cudaFree(gpuData->src_out);
		cudaFree(gpuData->ids_out);
		gpuDataList.pop_back();
	}

	while(cpuDataList.size() > 0) {
		CenterPostProcessingSharedData *cpuData = cpuDataList.back();
		cudaFreeHost(cpuData->scores);
		cudaFreeHost(cpuData->clses);
		cudaFreeHost(cpuData->bbx0);
		cudaFreeHost(cpuData->bby0);
		cudaFreeHost(cpuData->bbx1);
		cudaFreeHost(cpuData->bby1);
		cpuDataList.pop_back();
	}

	cudaFreeHost(ids);
	cudaFree(ids_d);

	while(target_coords_vec.size() > 0) {
		float * target_coords = target_coords_vec.back();
		free(target_coords);
		target_coords_vec.pop_back();
	}

	int batch = this->detection_num_vec[0].size();

	while(dets_vec.size() > 0)
	{
		Detection *det = dets_vec.back();
		deallocateDetectionBox(batch * NBOXES, det);
		dets_vec.pop_back();
	}

	delete dataset;
}


