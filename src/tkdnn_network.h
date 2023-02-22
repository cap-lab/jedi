
#ifndef TKDNN_NETWORK_H_
#define TKDNN_NETWORK_H_

#include <tkDNN/tkdnn.h>

#include "config_data.h"

#include "jedi_network.h"

class TkdnnNetwork : public IJediNetwork {
	public:
		TkdnnNetwork() {};
		tk::dnn::Network *net;
		//void createNetwork() override;
	private:
};


#endif
