import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from testFunctions import singleImageTest
from storeParameters import *




# Initialize the training parameters
from files import *
from runParameters import *



# Load the MNIST dataset
imgArray, labelsArray = loadDataset(testImages, testLabels)




# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights)



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

# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, testPerformanceFile)
