import numpy as np
import brian2 as b2
import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from trainFunctions import singleImageTraining

from utils import createDir
from storeParameters import storeParameters, storePerformace

from runParameters import *




# Load the MNIST dataset
imgArray, labelsArray = loadDataset(trainImages, trainLabels)



# Create the network data structure
network = createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename, 
		scaleFactors)


currentIndex  = 0
numberOfCycles = imgArray.shape[0]



# Measure the training starting time
startTimeTraining = timeit.default_timer()


while currentIndex < numberOfCycles:

	
	# Complete training cycle over a single image
	inputIntensity, currentIndex, accuracies = \
		singleImageTraining(
			trainDuration, 
			restTime, 
			imgArray[currentIndex], 
			network, 
			networkList, 
			currentSpikesCount, 
			prevSpikesCount, 
			countThreshold,
			inputIntensity, 
			currentIndex, 
			spikesEvolution, 
			updateInterval,
			printInterval, 
			startTimeTraining, 
			accuracies, 
			labelsArray,
			assignments, 
			startInputIntensity, 
			mode, 
			constSum
		)



# Create the directory in which to store the parameters and the performance
createDir(paramDir)

# Store the network parameters into NumPy files
storeParameters(network, networkList, assignments, weightFilename, 
		thetaFilename, assignmentsFile)

# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, trainPerformanceFile)
