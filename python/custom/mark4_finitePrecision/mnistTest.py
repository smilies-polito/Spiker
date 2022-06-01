import timeit

from createNetwork import createNetwork
from testFunctions import singleImageTest
from storeParameters import *
from utils import checkParallelism




# Initialize the training parameters
from files import *
from runParameters import *


# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, trainPrecision, rng)

checkParallelism(network["exc2exc1"]["weights"], weights_parallelism)


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
		constSums,
		rng,
		exp_shift,
		neuron_parallelism
	)

# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, testPerformanceFile)
