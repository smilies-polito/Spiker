import timeit
import sys
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

from createNetwork import createNetwork
from testFunctions import singleImageTest
from storeParameters import *
from utils import checkBitWidth

# Initialize the training parameters
from files import *
from runParameters import *

if mnistDir not in sys.path:
	sys.path.append(mnistDir)

from mnist import loadDataset



# Load the MNIST dataset
_ , labelsArray = loadDataset(testImages, testLabels)


# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, trainPrecision, rng)

checkBitWidth(network["exc2exc1"]["weights"], weights_bitWidth)

currentIndex = int(sys.argv[1])


# Measure the test starting time
startTimeTraining = timeit.default_timer()

# Complete test cycle over a single image
inputIntensity, currentIndex, accuracies, spikesMonitor, membraneMonitor, \
spikesCounter = \
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
		neuron_bitWidth,
		inputFilename
	)

with open(outSpikesFilename, "w") as spikes_fp:
	spikes_fp.write(str(list(spikesMonitor.astype(int))).replace(",",
		"").replace(" ", "\n")[1:-1])

with open(membraneFilename, "w") as membrane_fp:
	membrane_fp.write(str(list(membraneMonitor.astype(int))).replace(",",
		"").replace(" ", "\n")[1:-1])

with open(countersFilename, "w") as counter_fp:
	counter_fp.write(str(list(spikesCounter[0].astype(int))).replace(",",
		"").replace(" ", "\n")[1:-1])

# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, testPerformanceFile)
