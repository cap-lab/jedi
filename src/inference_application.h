
#ifndef INFERENCE_APPLICATION_H_
#define INFERENCE_APPLICATION_H_

#include <string>
#include <map>
#include <boost/function/function0.hpp>

#include <libconfig.h++>

#include "config_data.h"

#include "jedi_network.h"

#ifndef IN
	#define IN
#endif

#ifndef OUT
	#define OUT
#endif


class IInferenceApplication {
	public:
		IInferenceApplication() {};
		virtual ~IInferenceApplication() {};
		virtual void initializePreprocessing(std::string network_name, int maximum_batch_size, int thread_number) {};
		virtual void preprocessing(int thread_id, int input_tensor_index, const char *input_name, int sample_index, int batch_index, IN OUT float *input_buffer)  = 0;
		virtual void initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number) {};
		virtual void postprocessing1(int thread_id, int sample_index, IN float **output_buffers, int output_num, int batch) = 0;
		virtual void postprocessing2(int thread_id, int sample_index, int batch) = 0;
		//virtual std::string getInputModelName() = 0;
		//virtual tk::dnn::Network* createNetwork(ConfigInstance *basic_config_data) = 0;
		virtual IJediNetwork *createNetwork(ConfigInstance *basic_config_data) = 0;

		virtual void readCustomOptions(libconfig::Setting &setting) {};

		virtual void writeResultFile(std::string result_file_name) = 0;

		//virtual void initializeInference() = 0;
		//virtual void finalizeInference() {};
		//void registerModel(std::string model_type) { g_InputModelRegistry.create(model_type); }
	private:
		//IInputModel *input_model;
};


class AppRegistry
{
  typedef boost::function0<IInferenceApplication *> Creator;
  typedef std::map<std::string, Creator> Creators;
  Creators m_Creators;

public:
  template <class ConcreteType>
  void registerApp(const std::string &identifier)
  {
    ConstructorWrapper<ConcreteType> wrapper;
    Creator creator = wrapper;
    m_Creators[identifier] = creator;
  }

  IInferenceApplication *create(std::string &identifier)
  {
	  IInferenceApplication *result = nullptr;
    auto it = m_Creators.find(identifier);
    if (it != m_Creators.end())
    {
      result = it->second();
    }
    return result;
  }

protected:
  template<class ConcreteType>
  struct ConstructorWrapper
  {
	  IInferenceApplication *operator()() const { return new ConcreteType(); }
  };
};

extern AppRegistry g_AppRegistry;

template <typename T>
class AppRegisterar
{
public:
	AppRegisterar(std::string name) { g_AppRegistry.registerApp<T>(name); }
};

#define REGISTER_JEDI_APPLICATION(name) \
	static AppRegisterar<name> appRegistrar_##name = AppRegisterar<name>(#name)


#endif
