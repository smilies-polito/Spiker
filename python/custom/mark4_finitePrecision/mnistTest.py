import timeit
import sys

from mnist import loadDataset
from createNetwork import createNetwork
from testFunctions import singleImageTest
from storeParameters import *
from utils import checkParallelism




# Initialize the training parameters
from files import *
from runParameters import *



# Load the MNIST dataset
_ , labelsArray = loadDataset(testImages, testLabels)


# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, trainPrecision, rng)

checkParallelism(network["exc2exc1"]["weights"], weights_parallelism)

currentIndex = sys.argv[1]


# Measure the test starting time
startTimeTraining = timeit.default_timer()

# Complete test cycle over a single image
inputIntensity, currentIndex, accuracies = \
	singleImageTest(
		trainDuration,
		restTime,
		dt,
		None,
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
		inputFilename
	)

# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, testPerformanceFile)
