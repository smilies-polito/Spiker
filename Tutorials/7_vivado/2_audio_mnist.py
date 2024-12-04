import logging

from spikerplus.dataloaders import AudioMnistDL
from spikerplus import NetBuilder, Trainer, Optimizer, VhdlGenerator
from spikerplus.vhdl import write_vhdl, compile_vhdl, elaborate_vhdl

# Print progress at the different steps
logging.basicConfig(level=logging.INFO)

data_dir	= "../4_audio_mnist/AudioMnist/data"
batch_size	= 64

data_loader = AudioMnistDL(data_dir = data_dir)
train_loader, test_loader = data_loader.load(batch_size = 64)

# Extract number of timesteps of the input data (by default 73)
n_cycles = next(iter(train_loader))[0].shape[1]

# Extract number of inputs (by default 40)
n_inputs = next(iter(train_loader))[0].shape[2]

# Configure the Spiking Neural Network
net_dict = {

		"n_cycles"				: n_cycles,
		"n_inputs"				: n_inputs,

		"layer_0"	: {
			
			"neuron_model"		: "lif",
			"n_neurons"			: 128,
			"beta"				: 0.9375,
			"learn_beta"		: False,
			"threshold"			: 1.,
			"learn_threshold"	: False,
			"reset_mechanism"	: "subtract"
		},

		# Readout layer: leaky integrator
		"layer_1"	: {
			
			"neuron_model"		: "lif",
			"n_neurons"			: 10,
			"beta"				: 0.9375,
			"learn_beta"		: False,
			"threshold"			: 1.,
			"learn_threshold"	: False,
			# Don't reset the membrane: leaky integrator
			"reset_mechanism"	: "none"
		}
}

# Search ranges for the optimizer
optim_config = {

	"weights_bw"	: {
		"min"	: 4,
		"max"	: 10
	},

	"neurons_bw"	: {
		"min"	: 4,
		"max"	: 10
	},

	"fp_dec"	: {
		"min"	: 4,
		"max"	: 6
	}
}

# Instantiate network builder providing the network configuration
net_builder = NetBuilder(net_dict)

# Build snn model
snn = net_builder.build()

# Instantiate trainer
trainer = Trainer(snn)

# Train network and evaluate it on the test set
trainer.train(train_loader, test_loader, n_epochs = 1)

# Instantiate optimizer
opt = Optimizer(snn, net_dict, optim_config)

# Run grid search over provided quantization ranges
opt.optimize(test_loader)

# Ask the user to select the quantization values he/she prefers
optim_config = {}
optim_config["weights_bw"] 	= int(input(
	"Pick the best weights bitwidth: "))

optim_config["neurons_bw"]	= int(input(
	"Pick the best neurons bitwidth: "))

optim_config["fp_dec"]		= int(input(
	"Pick the best number of fixed point digits: "))

# Instantiate VHDL generateor
vhdl_generator = VhdlGenerator(snn, optim_config, functional = False)

# Generate VHDL
vhdl_snn  = vhdl_generator.generate(interface = True)

# Write all the VHDL sources
write_vhdl(vhdl_snn, output_dir = "SpikerAudioMnist")
compile_vhdl(vhdl_snn)
elaborate_vhdl(vhdl_snn)
