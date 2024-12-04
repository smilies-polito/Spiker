from spikerplus import NetBuilder, VhdlGenerator
from spikerplus.vhdl import write_vhdl, compile_vhdl, elaborate_vhdl

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
	"fp_dec"		: 4

}


net_builder = NetBuilder(net_dict)

snn = net_builder.build()

vhdl_generator = VhdlGenerator(snn, optim_config)

vhdl_snn  = vhdl_generator.generate()

write_vhdl(vhdl_snn, rm = True)
compile_vhdl(vhdl_snn)
elaborate_vhdl(vhdl_snn)
