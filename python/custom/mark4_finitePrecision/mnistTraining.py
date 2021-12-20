import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from trainFunctions import singleImageTraining
from storeParameters import *
from utils import createDir




# Initialize the training parameters
from files import *
from runParameters import *



# Load the MNIST dataset
imgArray, labelsArray = loadDataset(trainImages, trainLabels)




# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, trainPrecision)



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
			constSums,
			rng,
			exp_shift
		)


# Create the directory in which to store the parameters and the performance
createDir(paramDir)


# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, trainPerformanceFile)

# Store the network parameters into NumPy files
storeParameters(network, networkList, assignments, weightFilename, 
 		thresholdFilename, assignmentsFile)
