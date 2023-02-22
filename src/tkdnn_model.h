#ifndef TKDNN_MODEL_H_
#define TKDNN_MODEL_H_

#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <tkDNN/tkdnn.h>

#include "variable.h"
#include "config.h"
#include "stage.h"

#include "inference_application.h"

#include "model.h"

#include "tkdnn_network.h"

class TkdnnModel : public Model {
	public:
		tk::dnn::Network *net;
		TkdnnNetwork *tkdnn_network;
		TkdnnModel(ConfigData *config_data, int instance_id, IInferenceApplication *app) : Model(config_data, instance_id, app) {};
		~TkdnnModel() {};
		void initializeModel() override;
		void finalizeModel() override;
	private:
		std::vector<tk::dnn::NetworkRT *> netRTs;
		void getModelFileName(int curr, std::string &plan_file_name, int input_width, int input_height);
		void setDataType();
		void setDevice(int curr);
		void setMaxBatchSize();
		void createCalibrationTable(std::string plan_file_name, int iter, int start_index, int end_index);
		void readFromCalibrationTable(std::string basic_calibration_table, int start_index, int end_index, std::string out_calib_table, int device);
		int getLayerNumberFromCalibrationKey(std::string key);

};

#endif
