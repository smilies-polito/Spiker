import numpy as np
# import brian2 as b2
import timeit
import sys

from createNetwork import createNetwork
from trainTestFunctions import *
from utils import createDir, initAssignments
from runParameters import *

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


i = 0

startTimeTraining = timeit.default_timer()

numberOfCycles = imgArray.shape[0]

while i < numberOfCycles:

	inputIntensity, i, accuracies = trainTestCycle(imgArray[i], networkList, 
		network, trainDuration, restTime, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeTraining, accuracies, labelsArray, 
		assignements,inputIntensity, startInputIntensity, i, mode)


createDir(paramDir)

storeParameters(networkList, network, assignements, weightFilename, 
		thetaFilename, assignementsFilename)

storePerformace(startTimeTraining, accuracies, performanceFilename)
