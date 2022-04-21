
#ifndef INFERENCE_APPLICATION_H_
#define INFERENCE_APPLICATION_H_

#include <string>
#include <map>
#include <boost/function/function0.hpp>

#include <libconfig.h++>
#include <tkDNN/tkdnn.h>

#include "config_data.h"

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
		virtual void initializePostprocessing(std::string network_name, int maximum_batch_size, int thread_number) {};
		virtual tk::dnn::Network* createNetwork(ConfigInstance *basic_config_data) = 0;
		virtual void referNetworkRTInfo(int device_id, tk::dnn::NetworkRT *networkRT) {};
		virtual void readCustomOptions(libconfig::Setting &setting) {};
		virtual void preprocessing2(char *path, bool letter_box, int *pwidth, int *pheight, float *input_buffer) {};
		virtual void postprocessing2(int width, int height, char *input_data, float **output_buffers, int output_num, char *result_file_name) {};

	private:
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
