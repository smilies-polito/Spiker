import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from trainFunctions import singleImageTraining



print("\n1) Initialize training parameters\n")

# Initialize the training parameters
from files import *
from trainingParameters import *



print("\n2) Import the MNIST dataset\n")

# Load the MNIST dataset
imgArray, labelsArray = loadDataset(images, labels)



print("\n3) Create the network\n")

# Create the network data structure
network = createNetwork(networkList, weightFilename, thetaFilename, mode, 
			excDictList, inhDictList, scaleFactors, exc2inhWeights,
			inh2excWeights)



currentIndex = 0
numberOfCycles = imgArray.shape[0]


# Measure the training starting time
startTimeTraining = timeit.default_timer()



print("\n4) Start the training of the netwoek\n")
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

# storeParameters(networkList, network, assignements, weightFilename, 
# 		thetaFilename, assignementsFilename)
# 
# storePerformace(startTimeTraining, accuracies, performanceFilename)
