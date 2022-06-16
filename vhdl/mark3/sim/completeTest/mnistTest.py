import timeit
import sys
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp

from mnist import loadDataset
from testFunctions import singleImageTest
from storeParameters import *


# Initialize the training parameters
from files import *
from runParameters import *

sp.run(networkCompile)


# Load the MNIST dataset
imgArray , labelsArray = loadDataset(testImages, testLabels)


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
			networkScript,
			countersFilename,
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
			imageFilename,
			countBitWidth,
			pixelBitWidth
		)

# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, testPerformanceFile)
