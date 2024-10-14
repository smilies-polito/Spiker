from spiker import NetBuilder, VhdlGenerator
from spiker.vhdl.vhdl import write_file_all, fast_compile, elaborate

net_dict = {

		"n_cycles"				: 100,
		"n_inputs"				: 784,

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
	"neurons_bw"	: 6,
	"fp_dec"		: 4

}


net_builder = NetBuilder(net_dict)

snn = net_builder.build()

vhdl_generator = VhdlGenerator(snn, optim_config)

vhdl_snn  = vhdl_generator.generate()

write_file_all(vhdl_snn)
fast_compile(vhdl_snn)
elaborate(vhdl_snn)
