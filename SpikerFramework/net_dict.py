net_dict = {

		"n_cycles"				: 100,

		"n_inputs"				: 10,

		"layer_0"	: {
			
			"neuron_model"		: "if",
			"n_neurons"			: 4,
			"alpha"				: 0.8,
			"learn_alpha"		: False,
			"beta"				: 0.9,
			"learn_beta"		: False,
			"threshold"			: 1.,
			"learn_threshold"	: False
		},

		"layer_1"	: {
			
			"neuron_model"		: "lif",
			"n_neurons"			: 4,
			"alpha"				: 0.8,
			"learn_alpha"		: False,
			"beta"				: 0.9,
			"learn_beta"		: False,
			"threshold"			: 1.,
			"learn_threshold"	: False
		}
}
