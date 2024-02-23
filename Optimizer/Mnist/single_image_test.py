import timeit
import sys
import numpy as np

from snntorch import spikegen

from createNetwork import createNetwork
from poisson import imgToSpikeTrain
from network import run, rest
from storeParameters import *
from utils import checkBitWidth

from files import *
from runParameters import *
from bitWidths import *
from mnist import loadDataset

image_index	= 13
out_spikes_file	= "out_spikes_python.txt"

# Load the MNIST dataset
train_loader, test_loader = loadDataset(data_path, batch_size)

test_data = list(iter(test_loader))

# Create the network data structure
net = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, None,
			fixed_point_decimals, neuron_bitWidth, weights_bitWidth,
			trainPrecision, rng)


image = test_data[image_index][0].view(batch_size, -1).numpy()

label = int(test_data[image_index][1].int())

spikesTrains = imgToSpikeTrain(image, num_steps, rng)

_, _, out_spikes, _= run(net, networkList, spikesTrains,
		dt_tauDict, exp_shift, None, mode, None,
		neuron_bitWidth)

with open(out_spikes_file, "w") as fp:
	for output in out_spikes:
		fp.write((str(output.astype(int))[1:-1].replace(" ", "")))
		fp.write("\n")
