import timeit
import sys
import matplotlib.pyplot as plt
import numpy as np

from files import *

if pythonSrcDir not in sys.path:
	sys.path.insert(1, pythonSrcDir)

from mnist import loadDataset
from createNetwork import createNetwork
from testFunctions import singleImageTest
from storeParameters import *
from utils import checkParallelism
from runParameters import *

if bramInitDir not in sys.path:
	sys.path.insert(1, bramInitDir)

from bramInit import bramInit





# Load the MNIST dataset
imgArray, labelsArray = loadDataset(testImages, testLabels)


# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, trainPrecision, rng)

checkParallelism(network["exc2exc1"]["weights"], weights_parallelism)






currentIndex = int(sys.argv[1])


# Measure the test starting time
startTimeTraining = timeit.default_timer()

# Complete test cycle over a single image
inputIntensity, currentIndex, accuracies, spikesMonitor, membraneMonitor = \
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

with open(outSpikesFilename, "w") as spikes_fp:
	spikes_fp.write(str(list(spikesMonitor.astype(int))).replace(",",
		"")[1:-1])

with open(membraneFilename, "w") as membrane_fp:
	membrane_fp.write(str(list(membraneMonitor.astype(int))).replace(",",
		"")[1:-1])

# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, testPerformanceFile)






N_out = 16
N_neurons = 400

testImages = "../../../../mnist/t10k-images-idx3-ubyte"
testLabels = "../../../../mnist/t10k-labels-idx1-ubyte"

inputsFilename = "inputs.txt"
outputsFilename = "cntOut.txt"


sp.run("make clean", shell=True)

sp.run("./compile.sh")

# Load the MNIST dataset
imgArray, labelsArray = loadDataset(testImages, testLabels)

for image in imgArray:

	# Convert the image into spikes trains
	spikesTrains = imgToSpikeTrain(image, dt, testSteps, inputIntensity,
			rng)

	with open(inputsFilename, "w") as fp:

		for step in spikesTrains:

			inputs = np.array2string(step.astype(int),
					max_line_width = step.shape[0]*2+2)

			inputs = inputs[1:-1]

			inputs = inputs.replace(" ", "")

			fp.write(inputs)
			fp.write("\n")


	sp.run("./sim.sh")

	with open(outputsFilename) as fp:

		countersString = fp.readline()

		counters = [0 for _ in range(N_neurons)]

		for i in range(N_neurons):

			binaryString = countersString[i*N_out:(i+1)*N_out]

			counters[i] = int(binaryString, 2)

	print(counters)

	break


sp.run("make clean", shell=True)
