import numpy as np
import brian2 as b2
import timeit
import sys

from createNetwork import createNetwork
from runFunctions import singleImageRun

from utils import createDir
from storeParameters import storeParameters, storePerformace

from runParameters import *

if mnistDir not in sys.path:
	sys.path.append(mnistDir)

from mnist import loadDataset



# Load the MNIST dataset
imgArray, labelsArray = loadDataset(trainImages, trainLabels)



# Create the network data structure
network = createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename, 
		scaleFactors)


currentIndex  = 0
numberOfCycles = imgArray.shape[0]



# Measure the run starting time
startTimeRun = timeit.default_timer()


while currentIndex < numberOfCycles:

	
	# Complete cycle over a single image
	inputIntensity, currentIndex, accuracies = \
		singleImageRun(
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
			startTimeRun, 
			accuracies, 
			labelsArray,
			assignments, 
			startInputIntensity, 
			mode, 
			constSum
		)


if mode == "train":

	# Create the directory in which to store the parameters and the performance
	createDir(paramDir)

	# Store the network parameters into NumPy files
	storeParameters(network, networkList, assignments, weightFilename, 
			thetaFilename, assignmentsFile)


# Store the performance of the network into a text file
storePerformace(startTimeRun, accuracies, trainPerformanceFile)
