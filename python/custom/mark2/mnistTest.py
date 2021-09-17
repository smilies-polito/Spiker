import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from trainFunctions import singleImageTraining
from storeParameters import *




# Initialize the training parameters
from files import *
from runParameters import *



# Load the MNIST dataset
imgArray, labelsArray = loadDataset(testImages, testabels)




# Create the network data structure
network = createNetwork(networkList, weightFilename, thetaFilename, mode, 
			excDictList, inhDictList, scaleFactors, exc2inhWeights,
			inh2excWeights)



currentIndex = 0
numberOfCycles = imgArray.shape[0]


# Measure the test starting time
startTimeTraining = timeit.default_timer()



while currentIndex < numberOfCycles:

	# Complete test cycle over a single image
	inputIntensity, currentIndex, accuracies = \
		singleImageTest(
			trainDuration,
			restTime,
			dt,
			imgArray[currentIndex],
			pixelMin,
			pixelMax,
			network,
			networkList,
			dt_tauDict,
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
