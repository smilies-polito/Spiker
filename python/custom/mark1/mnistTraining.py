import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from trainFunctions import singleImageTraining



# Initialize the training parameters
from files import *
from trainingParameters import *


# Load the MNIST dataset
imgArray, labelsArray = loadDataset(images, labels)


# Create the network data structure
network = createNetwork(networkList, weightFilename, thetaFilename, mode, 
			excDictList, inhDictList, scaleFactors, exc2inhWeights,
			inh2excWeights)



currentIndex = 0
numberOfCycles = imgArray.shape[0]


# Measure the training starting time
startTimeTraining = timeit.default_timer()


while currentIndex < numberOfCycles:

	# Complete training cycle over a single image
	inputIntensity, i, accuracies = \
		singleImageTraining(
			trainDuration,
			restTime,
			dt,
			imgArray[currentIndex],
			pixelMin,
			pixelMax,
			network,
			networkList,
			dt_tauDict,
			stdpDict,
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
			constSums
		)

# storeParameters(networkList, network, assignements, weightFilename, 
# 		thetaFilename, assignementsFilename)
# 
# storePerformace(startTimeTraining, accuracies, performanceFilename)
