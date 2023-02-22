
#ifndef JEDI_NETWORK_H_
#define JEDI_NETWORK_H_

class IJediNetwork {
	public:
		IJediNetwork() {};
		virtual ~IJediNetwork() {};
		//virtual void destroyNetwork() = 0;
		//virtual std::string getInputModelName() = 0;
		//virtual networkT *createNetwork(ConfigInstance *basic_config_data) = 0;	
		//virtual void initializeInference() = 0;
		//virtual void finalizeInference() {};
		//void registerModel(std::string model_type) { g_InputModelRegistry.create(model_type); }
	private:
		//IInputModel *input_model;
};


#endif
