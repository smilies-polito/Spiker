import brian2 as b2
import timeit
import numpy as np
import sys

from poisson import imgToSpikeTrain
from parameters import *

locals().update(parametersDict)


def trainTestCycle(image, networkList, network, trainDuration, restTime, 
		spikesEvolution, updateInterval, printInterval, 
		currentSpikesCount, prevSpikesCount, startTimeTraining, 
		accuracies, labelsArray, assignments, inputIntensity, 
		startInputIntensity, currentIndex, mode):

	"""
	Perform a complete train/test pass over an image.

	INPUT:

		1) image: Numpy array of float values.

		2) networkList: list. Contains the network structure. Each
		element contain the number of neurons of the corresponding
		layer.

		3) network: Brian 2 Network object.

		4) trainDuration: temporal window associated to an input image

		5) restTime: temporal window associated to the rest after one
		pass over an image

		6) spikesEvolution: numpy array. Monitor the temporal evolution
		of the spikes.

		7) updateInterval: integer. Number of images after which the
		network parameters are updated and the accuracy is evaluated.

		8) printInterval: integer. Number of images after which the
		accuracy evolution is printed

		9) currentSpikesCount: integer. Number of output spikes
		generated at the previous training pass.

		10) prevSpikesCount: integer. Total number of spikes generated
		since the beginning of the training.

		11) startTimeTraining: float. Training/test initial time
		instant.

		12) accuracies: list. Evolution of the accuracy during the
		training/test.

		13) labelsArray: numpy array. Contains all the labels in the
		dataset.

		14) assignments: numpy array. Contains the labels currently
		associated to the output neurons.

		15) inputIntensity: float. Current value of the input scaling
		factor.

		16) startInputIntensity: float. Initial value of the input
		scaling factor.

		17) currentIndex: integer. Current image index.

		18) mode: string. Can be "train" or "test"

	OUTPUT:

		1) inputIntensity: float. Updated value of the input scaling
		factor.

		2) currentIndex: integer. Updated image index.

		3) accuracies: list. Updated evolution of the accuracy during
		the training/test.
		
	"""

	startTimeImage = timeit.default_timer()

	imgToSpikeTrain(network, image, inputIntensity)
	
	inputIntensity, currentIndex, accuracies = trainTestSingleImage(
		networkList, network, trainDuration, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeImage, startTimeTraining, accuracies,
		labelsArray, assignments, inputIntensity, startInputIntensity, 
		currentIndex,mode)


	imgToSpikeTrain(network, np.zeros(image.shape[0]), inputIntensity)

	b2.run(restTime)

	return inputIntensity, currentIndex, accuracies







	

def trainTestSingleImage(networkList, network, trainDuration, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeImage, startTimeTraining, accuracies, 
		labelsArray, assignments, inputIntensity, startInputIntensity, 
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
			assignments, startInputIntensity, currentIndex, mode)

	return inputIntensity, currentIndex, accuracies





def nextImage(networkList, spikesEvolution, updateInterval, printInterval, 
		currentSpikesCount, startTimeImage, startTimeTraining, 
		accuracies, labelsArray, assignments, startInputIntensity, 
		currentIndex, mode):

	spikesEvolution[currentIndex % updateInterval] = currentSpikesCount

	printProgress(currentIndex, printInterval, startTimeImage, 
			startTimeTraining)

	accuracies = computePerformances(currentIndex, updateInterval,
			networkList[-1], spikesEvolution,
			labelsArray[currentIndex - updateInterval :
			currentIndex], assignments, accuracies)

	if mode == "train":
		updateAssignements(currentIndex, updateInterval, 
				networkList[-1], spikesEvolution, 
				labelsArray[currentIndex - updateInterval
				: currentIndex], assignments)

	return startInputIntensity, currentIndex + 1, accuracies	




def repeatImage(inputIntensity, currentIndex):

	print("Increase inputIntensity from " + str(inputIntensity) + \
	" to " + str(inputIntensity + 1) + " for image " + str(currentIndex))

	return inputIntensity + 1





def updatePulsesCount(network, currentSpikesCount, prevSpikesCount):

	# Store the count of the spikes for the current image
	spikeMonitorCount = network.get_states(units=True, format='dict', 
				subexpressions=False, read_only_variables=True,
				level=0)["spikemonitor"]["count"]

	currentSpikesCount[:] = np.asarray(spikeMonitorCount) - prevSpikesCount
	prevSpikesCount[:] = np.asarray(spikeMonitorCount)







def updateAssignements(currentIndex, updateInterval, outputLayerSize,
			spikesEvolution, labelsSequence, assignments):

	maxCount = np.zeros(outputLayerSize)

	# Update every updateInterval
	if currentIndex % updateInterval == 0 and currentIndex > 0:

		# Loop over the 10 labels
		for label in range(10):

			# Total spikes count for the specific label
			labelSpikes = np.sum(spikesEvolution[labelsSequence ==
					label], axis=0)

			# Find where the spike count exceeds the current maximum
			whereMaxSpikes = labelSpikes > maxCount

			# Update the assignments	
			assignments[whereMaxSpikes] = label

			# Update the maxima
			maxCount[whereMaxSpikes] = labelSpikes[whereMaxSpikes]




def printProgress(currentIndex, printInterval, startTimeImage, 
			startTimeTraining):

	currentTime = timeit.default_timer()

	# Print every printInterval
	if currentIndex % printInterval == 0 and currentIndex > 0:
			
		progressString = "Analyzed images: " + str(currentIndex) + \
		". Time required for a single image: " + str(currentTime -
		startTimeImage) + "s. Total elapsed time: " + \
		str(timeit.default_timer() - startTimeTraining) + "s."

		print(progressString)





def computePerformances(currentIndex, updateInterval, outputLayerSize,
			spikesEvolution, labelsSequence, assignments, accuracies):

	maxCount = np.zeros(updateInterval)
	classification = -1*np.ones(updateInterval)

	# Update every updateInterval
	if currentIndex % updateInterval == 0 and currentIndex > 0:

		# Loop over the ten labels
		for label in range(10):

			# Consider only the neurons assigned to the current
			# label
			spikeCount = np.sum(spikesEvolution[:, assignments ==
					label], axis = 1)

			# Find where the neurons have generated max spikes
			whereMaxSpikes = spikeCount > maxCount

			# Update the classification along the updateInterval
			classification[whereMaxSpikes] = label

			# Update the maximum number of spikes for the label
			maxCount[whereMaxSpikes] = spikeCount[whereMaxSpikes]

		accuracies = updateAccuracy(classification, labelsSequence, accuracies)

	return accuracies






def updateAccuracy(classification, labelsSequence, accuracies):

	# Compute the number of correct classifications
	correct = np.where(classification == labelsSequence)[0].size

	# Compute the percentage of accuracy and add it to the list
	accuracies += ["{:.2f}".format(correct/classification.size*100) + "%"]
	
	# Print the accuracy
	accuracyString = "\nAccuracy: " + str(accuracies) + "\n"

	print(accuracyString)

	return accuracies
