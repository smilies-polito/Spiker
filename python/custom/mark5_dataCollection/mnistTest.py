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

weightsErrors = checkParallelism(network["exc2exc1"]["weights"], weights_parallelism,
		weightsErrors)


currentIndex = 0
numberOfCycles = 2#imgArray.shape[0]


maxInputSpikes = np.zeros(numberOfCycles+1)
maxOutputSpikes = np.zeros((numberOfCycles+1, len(networkList)-1))
cyclesCounter = np.zeros(numberOfCycles+1)


# Measure the test starting time
startTimeTraining = timeit.default_timer()



while currentIndex < numberOfCycles:

	# Complete test cycle over a single image
	inputIntensity, currentIndex, accuracies, \
	maxInputSpikes[currentIndex], maxOutputSpikes[currentIndex], \
	cyclesCounter[currentIndex], errorCounter = \
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
			maxOutputSpikes[currentIndex],
			errorCounter
		)


# Store the performance of the network into a text file
storePerformace(startTimeTraining, accuracies, testPerformanceFile)

# Store the array of maximum input spikes counts
storeArray(maxInputSpikesFile, maxInputSpikes)

# Store the array of maximum output spikes counts
storeArray(maxOutputSpikesFile, maxOutputSpikes)

# Store the array of cycles counts
storeArray(cyclesCounterFile, cyclesCounter)

decimalsString = "Decimals: " + str(fixed_point_decimals)
neuronParallelismString = "Parallelism: " + str(neuron_parallelism)
weightsParallelismString = "Weights parallelism: " + str(weights_parallelism)
weightsString = "Parallelism exceeded in " + str(weightsErrors) + " weights"
neuronString = "Parallelism exceeded in the neurons " + str(errorCounter) + \
			" times"

# Store the errors with a specific parallelism
with open(errorFile, 'a') as fp:
	fp.write(decimalsString)
	fp.write("\n")
	fp.write(neuronParallelismString)
	fp.write("\n")
	fp.write(weightsParallelismString)
	fp.write("\n")
	fp.write(weightsString)
	fp.write("\n")
	fp.write(neuronString)
	fp.write("\n\n")
