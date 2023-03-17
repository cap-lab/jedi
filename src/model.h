#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <boost/function/function3.hpp>

//#include <tkDNN/tkdnn.h>

#include "variable.h"
#include "config.h"
#include "stage.h"

#include "inference_application.h"

class Model {
	public:
		//InputDim input_dim;
		int total_input_size;

		std::vector<Stage *> stages;
        std::vector<std::map<std::string, void*>> all_stream_buffers;
        std::vector<std::map<std::string, bool*>> all_signals;


		std::vector<std::vector<float *>> net_input_buffers;
		std::vector<std::vector<float *>> net_output_buffers;
		std::vector<std::vector<bool *>> net_input_signals;
		std::vector<std::vector<bool *>> net_output_signals;
		int network_output_number;

		Model(ConfigData *config_data, int instance_id, IInferenceApplication *app);
		~Model();
		virtual void initializeModel() = 0;
		virtual void finalizeModel() = 0;
		void initializeBuffers();
		void finalizeBuffers();
		bool checkInferenceDone(int device_id, int stream_id);
		void infer(int device_id, int stream_id, int buffer_id);
		void waitUntilInferenceDone(int device_id, int stream_id);
		void waitUntilInputConsumed(int device_id, int stream_id);
		bool checkInputConsumed(int device_id, int stream_id);

		bool isPreprocessingRunnable(int buffer_id);
		bool isPostprocessingRunnable(int buffer_id);
		void updateInputSignals(int buffer_id, bool value);
		void updateOutputSignals(int buffer_id, bool value);

	protected:
		ConfigData *config_data;
		int instance_id;
		IInferenceApplication *app;

		void allocateStream();
		void allocateBuffer();
		void setBufferForStage();
		void deallocateBuffer();
		void deallocateStream();
		void* makeCUDAArray(int stage_id, int size);

		void allocateIOStreamBuffer(std::vector<std::pair<std::string, nvinfer1::Dims>> size_map, std::map<std::string, void*>& stream_buffers_map, std::vector<float *>& buffers, std::map<std::string, bool*>& signals_map, std::vector<bool*>& signals);
		void allocateStreamBuffer(int stage_id, int is_input_size_map, std::vector<std::pair<std::string, nvinfer1::Dims>> size_map, std::map<std::string, void*>& stream_buffers_map, std::map<std::string, bool*>& signals_map);
		void setBindingForContext(Stage *stage, int stream_id, int buffer_id);
		void setStreamBuffers(Stage *stage, int stream_id, int buffer_id);
};

class NetworkModelRegistry
{
	typedef boost::function3<Model *, ConfigData *, int, IInferenceApplication *> Creator;
  typedef std::map<std::string, Creator> Creators;
  Creators m_Creators;

public:
  template <class NetworkModelType>
  void registerNetworkModel(const std::string &identifier)
  {
    ConstructorWrapper<NetworkModelType> wrapper;
    Creator creator = wrapper;
    m_Creators[identifier] = creator;
	//printf("merong3\n");

  }

  Model *create(std::string &identifier, ConfigData *config_data, int instance_id, IInferenceApplication *app)
  {
	Model *result = nullptr;
    auto it = m_Creators.find(identifier);
    if (it != m_Creators.end())
    {
      result = it->second(config_data, instance_id, app);
    }
    return result;
  }

protected:
  template<class NetworkModelType>
  struct ConstructorWrapper
  {
	  Model *operator()(ConfigData *config_data, int instance_id, IInferenceApplication *app) const { return new NetworkModelType(config_data, instance_id, app); }
  };
};

extern NetworkModelRegistry g_NetworkModelRegistry;

template <typename T>
class NetworkModelRegisterar
{
public:
	NetworkModelRegisterar(std::string name) { g_NetworkModelRegistry.registerNetworkModel<T>(name); }
};

#define REGISTER_JEDI_NETWORK_MODEL(name) \
	static NetworkModelRegisterar<name> NetworkModelRegistrar_##name = NetworkModelRegisterar<name>(#name)


#endif
