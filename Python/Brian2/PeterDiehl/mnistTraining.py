import numpy as np
# import brian2 as b2
import timeit
import sys

from createNetwork import createNetwork
from utils import createDir, initAssignments
from storeParameters import storeParameters, storePerformance
from runParameters import *
from trainTestFunctions import *

from files import *

if mnistDir not in sys.path:
	sys.path.append(mnistDir)

from mnist import loadDataset


# Associate each output neuron to a label (-1 before training)
assignements = initAssignments(mode, networkList, assignementsFile)

# Import MNIST dataset into two numpy arrays
imgArray, labelsArray = loadDataset(images, labels)

# Create and initialize the Spiking Neural Network
network = createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename, 
		scaleFactors)

# Initialize image index
i = 0

# Number of images considered
numberOfCycles = imgArray.shape[0]

# Measure the beginning time
startTimeTraining = timeit.default_timer()

# Loop over the desired number of images
while i < numberOfCycles:

	# Perform a complete train/test cycle over the image
	inputIntensity, i, accuracies = trainTestCycle(imgArray[i], networkList, 
		network, trainDuration, restTime, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeTraining, accuracies, labelsArray, 
		assignements,inputIntensity, startInputIntensity, i, mode)


createDir(paramDir)

# Store the trained hyper-parameters
storeParameters(networkList, network, assignements, weightFilename, 
		thetaFilename, assignementsFilename)

# Store the results in terms of time and accuracy
storePerformance(startTimeTraining, accuracies, performanceFilename)
