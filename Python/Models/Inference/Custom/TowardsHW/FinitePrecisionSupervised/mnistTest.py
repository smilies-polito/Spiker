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


# Load the MNIST dataset
train_loader, test_loader = loadDataset(data_path, batch_size)

test_batch = iter(test_loader)

# Create the network data structure
net = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, None,
			fixed_point_decimals, neuron_bitWidth, weights_bitWidth,
			trainPrecision, rng)


checkBitWidth(net["exc2exc1"]["weights"], weights_bitWidth)


# Minibatch training loop
for test_data, test_targets in test_batch:

	acc = 0
	test_data = test_data.view(batch_size, -1)
	
	for i in range(test_data.size()[0]):

		image = test_data[i].numpy()
		label = int(test_targets[i].int())

		spikesTrains = imgToSpikeTrain(image, num_steps, rng)

		outputCounters, _, _ = run(net, networkList, spikesTrains,
				dt_tauDict, exp_shift, None, mode, None,
				neuron_bitWidth)


		rest(net, networkList)

		outputLabel = np.where(outputCounters[0] ==
				np.max(outputCounters[0]))[0][0]

		if outputLabel == label:
			acc += 1

	
	acc = acc / test_data.size()[0]
	print(f"Accuracy: {acc*100}%")
