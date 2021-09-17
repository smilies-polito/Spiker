import brian2 as b2
import timeit
import numpy as np
import sys


from poisson import *
from common import *

from equationsParameters import *
from neuronsParameters import *

# Add the parameters of the network to the local variables
locals().update(parametersDict)



def singleImageTest(image, networkList, network, trainDuration, restTime, 
		spikesEvolution, updateInterval, printInterval, 
		currentSpikesCount, prevSpikesCount, startTimeTraining, 
		accuracies, labelsArray, assignements, inputIntensity, 
		startInputIntensity, currentIndex, mode, constSum):

	startTimeImage = timeit.default_timer()

	imgToSpikeTrain(network, image, inputIntensity)
	
	inputIntensity, currentIndex, accuracies = test(
		networkList, network, trainDuration, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeImage, startTimeTraining, accuracies,
		labelsArray, assignements, inputIntensity, startInputIntensity, 
		currentIndex,mode)


	imgToSpikeTrain(network, np.zeros(image.shape[0]), inputIntensity)

	b2.run(restTime)

	return inputIntensity, currentIndex, accuracies





	

def test(networkList, network, trainDuration, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeImage, startTimeTraining, accuracies, 
		labelsArray, assignements, inputIntensity, startInputIntensity, 
		currentIndex, mode):

	network.run(trainDuration)


	updatePulsesCount(network, currentSpikesCount, prevSpikesCount)


	if np.sum(currentSpikesCount) < 5:

		inputIntensity = repeatImage(inputIntensity, currentIndex)

	else:

		inputIntensity, currentIndex, accuracies = nextImage(
			networkList, spikesEvolution, updateInterval, 
			printInterval, currentSpikesCount, startTimeImage,
			startTimeTraining, accuracies, labelsArray, 
			assignements, startInputIntensity, currentIndex, mode)

	return inputIntensity, currentIndex, accuracies






def normalizeNetWeights(network, networkList, constSum):

	normalizeLayerWeights(network, "poisson2exc", networkList[0],
				networkList[1], constSum)

	for i in range(2, len(networkList)):

		normalizeLayerWeights(network, "exc2exc" + str(i-1),
				networkList[i-1], networkList[i], constSum)




def normalizeLayerWeights(network, connectionName, inputLayerSize, excLayerSize,
			constSum):

	connection = network.get_states()

	newWeights = normalizeWeights(connection[connectionName]["w"], 
			inputLayerSize, excLayerSize, constSum)

	values = {
		connectionName : {
			"w" : newWeights
		}
	}

	network.set_states(values)

	




def normalizeWeights(weightsArray, inputLayerSize, excLayerSize, constSum):

	newWeights = np.copy(weightsArray)

	newWeights = np.reshape(newWeights, (inputLayerSize, excLayerSize))

	normFactors = newWeights.sum(axis=0)

	normFactors[normFactors == 0] = 1

	normFactors = constSum/normFactors

	newWeights = newWeights*normFactors

	return np.reshape(newWeights, inputLayerSize*excLayerSize)
