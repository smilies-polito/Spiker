import logging
from spikerplus import NetBuilder

# Print result of network build
logging.basicConfig(level=logging.INFO)

net_dict = {

		"n_cycles"				: 10,
		"n_inputs"				: 4,

		"layer_0"	: {
			
			"neuron_model"		: "lif",
			"n_neurons"			: 3,
			"alpha"				: None,
			"learn_alpha"		: False,
			"beta"				: 0.9375,
			"learn_beta"		: False,
			"threshold"			: 1.,
			"learn_threshold"	: False,
			"reset_mechanism"	: "subtract"
		},

		"layer_1"	: {
			
			"neuron_model"		: "lif",
			"n_neurons"			: 2,
			"alpha"				: None,
			"learn_alpha"		: False,
			"beta"				: 0.9375,
			"learn_beta"		: False,
			"threshold"			: 1.,
			"learn_threshold"	: False,
			"reset_mechanism"	: "none"
		}
}

optim_config = {

	"weights_bw"	: 4,
	"neurons_bw"	: 6,
	"fp_dec"		: 3

}

net_builder = NetBuilder(net_dict)

snn = net_builder.build()
