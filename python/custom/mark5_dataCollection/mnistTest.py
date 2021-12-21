import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from testFunctions import singleImageTest
from storeParameters import *
from utils import checkParallelism




# Initialize the training parameters
from files import *
from runParameters import *



# Load the MNIST dataset
imgArray, labelsArray = loadDataset(testImages, testLabels)




# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, trainPrecision, rng)

checkParallelism(network["exc2exc1"]["weights"], weights_parallelism)


currentIndex = 0
numberOfCycles = imgArray.shape[0]


maxInputSpikes = np.zeros(numberOfCycles+1)
maxOutputSpikes = np.zeros((numberOfCycles+1, len(networkList)-1))


# Measure the test starting time
startTimeTraining = timeit.default_timer()



while currentIndex < numberOfCycles:

	# Complete test cycle over a single image
	inputIntensity, currentIndex, accuracies, maxInputSpikes[currentIndex], \
	maxOutputSpikes[currentIndex] = \
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
			constSums,
			rng,
			exp_shift,
			neuron_parallelism,
			maxInputSpikes[currentIndex], 
			maxOutputSpikes[currentIndex]
		)


# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, testPerformanceFile)

# Store the array of maximum input spikes counts
storeArray(maxInputSpikesFile, maxInputSpikes)

# Store the array of maximum output spikes counts
storeArray(maxOutputSpikesFile, maxOutputSpikes)
