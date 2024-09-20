import logging
from torch.utils.data import DataLoader, random_split

from spiker import NetBuilder, Optimizer

import sys

audio_mnist_dir = "../../SnnModels/SnnTorch/AudioMnist/"

if audio_mnist_dir not in sys.path:
	sys.path.append(audio_mnist_dir)

from audio_mnist import MelFilterbank, CustomDataset

logging.basicConfig(level=logging.INFO)

root_dir	= audio_mnist_dir + "/Data"
batch_size	= 64

sample_rate	= 48e3

# Short Term Fourier Transform (STFT) window
fft_window	= 25e-3 # s

# Step from one window to the other (controls overlap)
hop_length_s	= 10e-3 #s

# Number of input channels: filters in the mel bank
n_mels		= 40

# Spiking threshold
spiking_thresh 	= 0.9

transform = MelFilterbank(
	sample_rate 	= sample_rate,
	fft_window 	= fft_window,
	hop_length_s	= hop_length_s,
	n_mels 		= n_mels,
	db 		= True,
	normalize	= True,
	spikify		= True,
	spiking_thresh	= spiking_thresh
)

dataset = CustomDataset(
	root_dir	= root_dir,
	transform	= transform
)

# Train/test split
train_size 		= int(0.8 * len(dataset))
test_size		= len(dataset) - train_size

# Split the dataset into training and validation sets
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, 
	batch_size	= batch_size,
	shuffle		= True,
	num_workers	= 4,
	drop_last 	= True
)

test_loader = DataLoader(test_set, 
	batch_size	= batch_size,
	shuffle		= True,
	num_workers	= 4,
	drop_last 	= True
)

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

opt = Optimizer(snn, net_dict, optim_config)

opt.optimize(test_loader)
