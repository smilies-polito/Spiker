import logging

from spiker import NetBuilder, Trainer
from Dataloaders.audio_mnist_dl import AudioMnistDL

logging.basicConfig(level=logging.INFO)

data_dir	= "AudioMnist/data"
batch_size	= 64

data_loader = AudioMnistDL(data_dir = data_dir)
train_loader, test_loader = data_loader.load(batch_size = 64)

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

	"weights_bw"	: {
		"min"	: 5,
		"max"	: 6
	},

	"neurons_bw"	: {
		"min"	: 5,
		"max"	: 6
	},

	"fp_dec"	: {
		"min"	: 2,
		"max"	: 3
	}
}


logging.basicConfig(level=logging.INFO)

net_builder = NetBuilder(net_dict)

snn = net_builder.build()

trainer = Trainer(snn)

trainer.train(train_loader, test_loader)
