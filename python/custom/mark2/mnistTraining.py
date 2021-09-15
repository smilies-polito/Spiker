import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from trainFunctions import singleImageTraining
from storeParameters import *




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
	inputIntensity, currentIndex, accuracies = \
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


# Store the network parameters into NumPy files
storeParameters(network, networkList, assignments, weightFilename, 
 		thetaFilename, assignmentsFile)
 
# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, performanceFile)
