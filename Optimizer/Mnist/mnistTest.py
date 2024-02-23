import timeit
import sys
import numpy as np

from snntorch import spikegen

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from createNetwork import createNetwork
from poisson import imgToSpikeTrain
from network import run, rest
from storeParameters import *
from utils import checkBitWidth

from runParameters import *
from bitWidths import *

image_height		= 28
image_width		= 28

# Directory in which parameters and performance of the network are stored
paramDir = "./Parameters128"

# Name of the parameters files
weightFilename = paramDir + "/weights"
thresholdFilename = paramDir + "/thresholds"
assignmentsFile = paramDir + "/assignments.npy"

# Name of the performance files
trainPerformanceFile = paramDir + "/trainPerformance.txt"
testPerformanceFile = paramDir + "/testPerformance.txt"

data_dir ='./data/mnist'

# Define a transform
transform = transforms.Compose([
	transforms.Resize((image_width, image_height)),
	transforms.Grayscale(),
	transforms.ToTensor(),
	transforms.Normalize((0,), (1,))]
)

test_set = datasets.MNIST(root=data_dir, train=False, download=True,
		transform=transform)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
		drop_last=True)


test_batch = iter(test_loader)

# Create the network data structure
net = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, None,
			fixed_point_decimals, neuron_bitWidth, weights_bitWidth,
			trainPrecision, rng)


# Minibatch training loop
for test_data, test_targets in test_batch:

	acc = 0
	test_data = test_data.view(batch_size, -1)
	
	for i in range(test_data.size()[0]):

		image = test_data[i].numpy()
		label = int(test_targets[i].int())

		spikesTrains = imgToSpikeTrain(image, num_steps, rng)

		outputCounters, _, _, _= run(net, networkList, spikesTrains,
				dt_tauDict, exp_shift, None, mode, None,
				neuron_bitWidth)


		rest(net, networkList)

		outputLabel = np.where(outputCounters[0] ==
				np.max(outputCounters[0]))[0][0]

		if outputLabel == label:
			acc += 1

	
	acc = acc / test_data.size()[0]
	print(f"Accuracy: {acc*100}%")
