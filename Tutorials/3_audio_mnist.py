import logging

from spiker import NetBuilder, Trainer, Optimizer, VhdlGenerator
from spiker.vhdl import write_vhdl, compile_vhdl, elaborate_vhdl

from Dataloaders.audio_mnist_dl import AudioMnistDL

logging.basicConfig(level=logging.INFO)

data_dir	= "AudioMnist/data"
batch_size	= 64

data_loader = AudioMnistDL(data_dir = data_dir)
train_loader, test_loader = data_loader.load(batch_size = 64)

n_cycles = next(iter(train_loader))[0].shape[1]
n_inputs = next(iter(train_loader))[0].shape[2]

net_dict = {

		"n_cycles"				: n_cycles,
		"n_inputs"				: n_inputs,

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

	"weights_bw"	: {
		"min"	: 7,
		"max"	: 10
	},

	"neurons_bw"	: {
		"min"	: 8,
		"max"	: 10
	},

	"fp_dec"	: {
		"min"	: 4,
		"max"	: 6
	}
}

net_builder = NetBuilder(net_dict)

snn = net_builder.build()

trainer = Trainer(snn)

trainer.train(train_loader, test_loader, n_epochs = 1)

opt = Optimizer(snn, net_dict, optim_config)

opt.optimize(test_loader)

optim_config = {}
optim_config["weights_bw"] 	= int(input(
	"Pick the best weights bitwidth: "))

optim_config["neurons_bw"]	= int(input(
	"Pick the best neurons bitwidth: "))

optim_config["fp_dec"]		= int(input(
	"Pick the best number of fixed point digits: "))

vhdl_generator = VhdlGenerator(snn, optim_config)

vhdl_snn  = vhdl_generator.generate()

write_vhdl(vhdl_snn, output_dir = "SpikerAudioMnist")
compile_vhdl(vhdl_snn, output_dir = "SpikerAudioMnist")
elaborate_vhdl(vhdl_snn, output_dir = "SpikerAudioMnist")
