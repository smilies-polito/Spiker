import timeit
import sys
import numpy as np

from snntorch import spikegen

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from network import run, rest, createNetwork
from utils import fixedPoint

from bitwidths import *

def imgToSpikeTrain(image, num_steps, rng):

	""" 
	Convert a black and white image into spike trains using the Poisson
	method.

	INPUT:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) dt: float. Time step duration, expressed in milliseconds. 

		3) trainingSteps: total amount of time steps associated to the
		pixels" spikes trains.

		4) inputIntensity: current value of the pixel"s intensity.

		5) rng: NumPy random generator.


	OUTPUT:

		Two-dimensional boolean NumPy array. Each row corresponds to a
		time step. Each column corresponds to a pixel.
	"""

	# Create two-dimensional array of random values
	random2D = rng.uniform(size = (num_steps, image.shape[0]))

	# Convert the image into spikes trains
	return poisson(image, random2D)


def poisson(image, random2D):

	""" 
	Poisson convertion of the numerical values of the pixels into spike
	trains. 

	INPUT PARAMETERS:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) dt: time step duration, expressed in milliseconds. 

		3) random2D: teo-dimensional NumPy array containing random
		values between pixelMin and pixelMax.

		4) inputIntensity: current value of the pixel"s intensity.


	OUTPUT:

		Boolean two-dimensional array containing one spikes"
		train for each pixel.  
	"""

	# Create the boolean array of spikes with Poisson distribution
	return image[:] > random2D



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

image_height		= 28
image_width		= 28

# dataloader arguments
batch_size = 256

dtype = torch.float

# Network Architecture
num_inputs = image_width*image_height
num_hidden = 128
num_outputs = 10

# Temporal Dynamics
num_steps = 100
beta = 0.9375
tauExc = 100


num_epochs = 30
loss_hist = []
test_loss_hist = []
counter = 0

# List of layer sizes
networkList = [784, 128, 10]

mode = "test"
trainPrecision = "float"

# Excitatory layer
excDict = {			# Shifted values to minimize the interval
				# ----------------------------------------------
				# |Original|	|  Ref  |	|Shift |
				# |--------|----|-------|-------|------|
	"vRest"		: 0,	# | -65.0  | =	| -65.0 | + 	| 0.0  | mV
	"vReset"	: 0,	# | -60.0  | =	| -65.0 | + 	| 5.0  | mV
	"vThresh0"	: 1,	# | -52.0  | =	| -65.0 | + 	| 13.0 | mV
	"vThreshPlus"	: 0.05,	# |  0.05  | =	|   -   | + 	|  -   | mV
}


# Finite precision
excDict["vRest"] = fixedPoint(excDict["vRest"], fixed_point_decimals,
		neuron_bitWidth[0])
excDict["vReset"] = fixedPoint(excDict["vReset"], fixed_point_decimals,
		neuron_bitWidth[0])
excDict["vThresh0"] = fixedPoint(excDict["vThresh0"], fixed_point_decimals,
		neuron_bitWidth[0])
excDict["vThreshPlus"] = fixedPoint(excDict["vThreshPlus"],
				fixed_point_decimals, neuron_bitWidth[0])

# List of dictionaries of parameters for the layers
excDictList = [excDict] * (len(networkList) - 1)

# Training and resting periods in milliseconds
trainDuration = 350	# ms

# Array of scale factors for the random generation of the weights
scaleFactors = np.ones(len(networkList) - 1)

# Time step duration in milliseconds
dt = 2**-4 * tauExc		# ms


# Exponential time constants
dt_tauDict = {
	"exc" 		: dt/tauExc,
}

# Update and print intervals expressed in number of images
updateInterval = 10
printInterval = 250

# Initial intensity of the input pixels
startInputIntensity = 2.
inputIntensity = startInputIntensity


# Initialize history of accuracies
accuracies = []

# Initialize history of spikes
spikesEvolution = np.zeros((updateInterval, networkList[-1]))


# Minimum acceptable number of output spikes generated during the training.
countThreshold = 5

# NumPy default random generator.
rng = np.random.default_rng()



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
