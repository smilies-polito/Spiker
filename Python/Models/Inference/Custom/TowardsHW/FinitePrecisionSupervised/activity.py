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

np.set_printoptions(threshold=np.inf)


# Load the MNIST dataset
train_loader, test_loader = loadDataset(data_path, batch_size)

test_batch = iter(test_loader)

# Create the network data structure
net = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, None,
			fixed_point_decimals, neuron_bitWidth, weights_bitWidth,
			trainPrecision, rng)


print("Shape: ", net["exc2exc1"]["weights"].shape)
print("Average: ", np.average(net["exc2exc1"]["weights"]))
print("Zeros:" , np.sum(net["exc2exc1"]["weights"] ==
	0)/(net["exc2exc2"]["weights"].shape[0]*
	net["exc2exc2"]["weights"].shape[1]))

tot = 0

s0 = ""
for j in range(128):
		if net["exc2exc1"]["weights"][j, 0] > 0:
			s0 += "{0:04b}".format(net["exc2exc1"]["weights"][j, 0])
		else:
			s0 += "{0:04b}".format(2**weights_bitWidth[0] -
					net["exc2exc1"]["weights"][j, 0])

for i in range(1, 784):

	s1 = ""
	for j in range(128):
		if net["exc2exc1"]["weights"][j, i] > 0:
			s1 += "{0:04b}".format(net["exc2exc1"]["weights"][j, i])
		else:
			s1 += "{0:04b}".format(2**weights_bitWidth[0] -
					net["exc2exc1"]["weights"][j, i])

	for j in range(128*weights_bitWidth[0]):
		if s1[j] != s0[j]:
			tot += 1


	s0 = s1


tot = tot / (net["exc2exc1"]["weights"].shape[0]*
		net["exc2exc1"]["weights"].shape[1]*weights_bitWidth[0])


print("Total switching activity hidden layer 1: ", tot)


s0 = ""
for j in range(10):
		if net["exc2exc2"]["weights"][j, 0] > 0:
			s0 += "{0:04b}".format(net["exc2exc2"]["weights"][j, 0])
		else:
			s0 += "{0:04b}".format(2**weights_bitWidth[0] -
					net["exc2exc2"]["weights"][j, 0])

for i in range(1, 128):

	s1 = ""
	for j in range(10):
		if net["exc2exc2"]["weights"][j, i] > 0:
			s1 += "{0:04b}".format(net["exc2exc2"]["weights"][j, i])
		else:
			s1 += "{0:04b}".format(2**weights_bitWidth[0] -
					net["exc2exc2"]["weights"][j, i])

	for j in range(10*weights_bitWidth[0]):
		if s1[j] != s0[j]:
			tot += 1


	s0 = s1


tot = tot / (net["exc2exc2"]["weights"].shape[0]*
		net["exc2exc2"]["weights"].shape[1]*weights_bitWidth[0])


print("Total switching activity in output_layer: ", tot)


input_spiking_activity = 0
hidden_spiking_activity = 0
output_spiking_activity = 0

count = 0

# Minibatch training loop
for test_data, test_targets in test_batch:

	acc = 0
	test_data = test_data.view(batch_size, -1)
	
	for i in range(test_data.size()[0]):

		image = test_data[i].numpy()
		label = int(test_targets[i].int())

		spikesTrains = imgToSpikeTrain(image, num_steps, rng)

		_, spikesMonitor_0, spikesMonitor_1, _ = run(net, networkList, spikesTrains,
				dt_tauDict, exp_shift, None, mode, None,
				neuron_bitWidth)


		rest(net, networkList)

		spikesMonitor_in = np.sum(np.array(spikesTrains))
		spikesMonitor_0 = np.sum(np.array(spikesMonitor_0))
		spikesMonitor_1 = np.sum(np.array(spikesMonitor_1))

		input_spiking_activity += spikesMonitor_in / (num_inputs*num_steps)
		hidden_spiking_activity += spikesMonitor_0 / (num_hidden*num_steps)
		output_spiking_activity += spikesMonitor_1 / (num_outputs*num_steps)

		count += 1

input_spiking_activity = input_spiking_activity / count
hidden_spiking_activity = hidden_spiking_activity / count
output_spiking_activity = output_spiking_activity / count

print("Input spike activity: ", input_spiking_activity)
print("Hidden spike activity: ", hidden_spiking_activity)
print("Output spike activity: ", output_spiking_activity)
