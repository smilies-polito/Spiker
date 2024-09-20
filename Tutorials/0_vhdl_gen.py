from spiker import NetBuilder, VhdlGenerator

net_dict = {

		"n_cycles"				: 73,
		"n_inputs"				: 40,

		"layer_0"	: {
			
			"neuron_model"		: "lif",
			"n_neurons"			: 128,
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
			"n_neurons"			: 10,
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
	"neurons_bw"	: 8,
	"fp_dec"		: 3

}


net_builder = NetBuilder(net_dict)

snn = net_builder.build()

vhdl_generator = VhdlGenerator(snn, optim_config)

vhdl_generator.generate()
