import sys
import numpy as np
import pickle

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
_, test_loader = loadDataset(data_path, batch_size)


#acc_vs_shift = []
#
# weights_bitWidth	= 64
# neuron_bitWidth		= 64
# exp_shift		= 10
# 
# for fixed_point_decimals in range(2):
#
#	test_batch = iter(test_loader)
# 
# 	# Create the network data structure
# 	net = createNetwork(
# 		networkList,
# 		weightFilename,
# 		thresholdFilename,
# 		mode, 
# 		excDictList,
# 		scaleFactors,
# 		None,
# 		fixed_point_decimals,
# 		neuron_bitWidth,
# 		weights_bitWidth,
# 		trainPrecision, 
# 		rng
# 	)
# 
# 	count = 0
# 	acc_average = 0
# 
# 	# Minibatch training loop
# 	for test_data, test_targets in test_batch:
# 
# 		acc = 0
# 		test_data = test_data.view(batch_size, -1)
# 		
# 		for i in range(test_data.size()[0]):
# 
# 			image = test_data[i]
# 			label = int(test_targets[i].int())
# 
# 			#input_spikes = imgToSpikeTrain(image.numpy(), num_steps, rng)
# 			input_spikes = spikegen.rate(image, num_steps = num_steps,
# 					gain = 1)
# 
# 			input_spikes = input_spikes.numpy().astype(bool)
# 
# 			outputCounters, _, _ = run(net, networkList, input_spikes,
# 					dt_tauDict, exp_shift, None, mode, None,
# 					neuron_bitWidth)
# 
# 
# 			rest(net, networkList)
# 
# 			outputLabel = np.where(outputCounters[0] ==
# 					np.max(outputCounters[0]))[0][0]
# 
# 			if outputLabel == label:
# 				acc += 1
# 
# 		
# 		acc_average += acc / test_data.size()[0]
# 
# 		count += 1
# 
# 	acc_average /= count
# 	acc_vs_shift.append(acc_average)
# 
# 
# with open("acc_vs_shift.npy", "wb") as fp:
# 	np.save(fp, np.array(acc_vs_shift))



# acc_vs_exp_shift = []
# 
# weights_bitWidth	= 64
# neuron_bitWidth		= 64
# fixed_point_decimals	= 32
# 
# 
# # Create the network data structure
# net = createNetwork(
# 	networkList,
# 	weightFilename,
# 	thresholdFilename,
# 	mode, 
# 	excDictList,
# 	scaleFactors,
# 	None,
# 	fixed_point_decimals,
# 	neuron_bitWidth,
# 	weights_bitWidth,
# 	trainPrecision, 
# 	rng
# )
# 
# for exp_shift in range(64):
# 
# 	test_batch = iter(test_loader)
# 
# 	count = 0
# 	acc_average = 0
# 
# 	# Minibatch training loop
# 	for test_data, test_targets in test_batch:
# 
# 		acc = 0
# 		test_data = test_data.view(batch_size, -1)
# 		
# 		for i in range(test_data.size()[0]):
# 
# 			image = test_data[i]
# 			label = int(test_targets[i].int())
# 
# 			#input_spikes = imgToSpikeTrain(image.numpy(), num_steps, rng)
# 			input_spikes = spikegen.rate(image, num_steps = num_steps,
# 					gain = 1)
# 
# 			input_spikes = input_spikes.numpy().astype(bool)
# 
# 			outputCounters, _, _ = run(net, networkList, input_spikes,
# 					dt_tauDict, exp_shift, None, mode, None,
# 					neuron_bitWidth)
# 
# 
# 			rest(net, networkList)
# 
# 			outputLabel = np.where(outputCounters[0] ==
# 					np.max(outputCounters[0]))[0][0]
# 
# 			if outputLabel == label:
# 				acc += 1
# 
# 		
# 		acc_average += acc / test_data.size()[0]
# 
# 		count += 1
# 
# 	acc_average /= count
# 	acc_vs_exp_shift.append(acc_average)
# 
# 
# with open("acc_vs_exp_shift.npy", "wb") as fp:
# 	np.save(fp, np.array(acc_vs_exp_shift))



acc_vs_shift = {}

neuron_bitWidth		= [64, 64]
exp_shift		= 10
w2_bw = 64

for fixed_point_decimals in range(4, 9):

	fp_decimals = "fp_decimals_" + str(fixed_point_decimals)

	acc_vs_weights = []

	for w1_bw in range(63, -1, -1):

		weights_bitWidth = [w1_bw, w2_bw]

		test_batch = iter(test_loader)

		# Create the network data structure
		net = createNetwork(
			networkList,
			weightFilename,
			thresholdFilename,
			mode, 
			excDictList,
			scaleFactors,
			None,
			fixed_point_decimals,
			neuron_bitWidth,
			weights_bitWidth,
			trainPrecision, 
			rng
		)

		count = 0
		acc_average = 0

		# Minibatch training loop
		batch_count = 0
		max_count = 2
		for test_data, test_targets in test_batch:

			if batch_count > max_count:
				break

			acc = 0
			test_data = test_data.view(batch_size, -1)
			
			for i in range(test_data.size()[0]):

				image = test_data[i]
				label = int(test_targets[i].int())

				#input_spikes = imgToSpikeTrain(image.numpy(), num_steps, rng)
				input_spikes = spikegen.rate(image, num_steps = num_steps,
						gain = 1)

				input_spikes = input_spikes.numpy().astype(bool)

				outputCounters, _, _ = run(net, networkList, input_spikes,
						dt_tauDict, exp_shift, None, mode, None,
						neuron_bitWidth)


				rest(net, networkList)

				outputLabel = np.where(outputCounters[0] ==
						np.max(outputCounters[0]))[0][0]

				if outputLabel == label:
					acc += 1

			
			acc_average += acc / test_data.size()[0]

			count += 1
			batch_count += 1

		acc_average /= count
		acc_vs_weights.append(acc_average)

	acc_vs_shift[fp_decimals] = acc_vs_weights

with open("acc_vs_weights.pkl", "wb") as fp:
	pickle.dump(acc_vs_shift, fp)


# fixed_point_decimals	= 4
# weights_bitWidth	= 4
# exp_shift		= 10
# 
# acc_vs_bw = []
# 
# for neuron_bitWidth in range(1, 64):
# 
# 	test_batch = iter(test_loader)
# 
# 	# Create the network data structure
# 	net = createNetwork(
# 		networkList,
# 		weightFilename,
# 		thresholdFilename,
# 		mode, 
# 		excDictList,
# 		scaleFactors,
# 		None,
# 		fixed_point_decimals,
# 		neuron_bitWidth,
# 		weights_bitWidth,
# 		trainPrecision, 
# 		rng
# 	)
# 
# 	count = 0
# 	acc_average = 0
# 
# 	# Minibatch training loop
# 	for test_data, test_targets in test_batch:
# 
# 		acc = 0
# 		test_data = test_data.view(batch_size, -1)
# 		
# 		for i in range(test_data.size()[0]):
# 
# 			image = test_data[i]
# 			label = int(test_targets[i].int())
# 
# 			#input_spikes = imgToSpikeTrain(image.numpy(), num_steps, rng)
# 			input_spikes = spikegen.rate(image, num_steps = num_steps,
# 					gain = 1)
# 
# 			input_spikes = input_spikes.numpy().astype(bool)
# 
# 			outputCounters, _, _ = run(net, networkList, input_spikes,
# 					dt_tauDict, exp_shift, None, mode, None,
# 					neuron_bitWidth)
# 
# 			rest(net, networkList)
# 
# 			outputLabel = np.where(outputCounters[0] ==
# 					np.max(outputCounters[0]))[0][0]
# 
# 			if outputLabel == label:
# 				acc += 1
# 
# 		
# 		acc_average += acc / test_data.size()[0]
# 
# 		count += 1
# 
# 		acc_average /= count
# 		acc_vs_bw.append(acc_average)
# 
# with open("acc_vs_bw.npy", "wb") as fp:
# 	np.save(fp, np.array(acc_vs_bw))
