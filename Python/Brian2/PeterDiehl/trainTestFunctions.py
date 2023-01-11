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

	# Beginning time for the training/test pass on the single image
	startTimeImage = timeit.default_timer()

	# Convert the input image into trains of spikes
	imgToSpikeTrain(network, image, inputIntensity)
	
	# Train/test cycle on the single image
	inputIntensity, currentIndex, accuracies = trainTestSingleImage(
		networkList, network, trainDuration, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeImage, startTimeTraining, accuracies,
		labelsArray, assignments, inputIntensity, startInputIntensity, 
		currentIndex,mode)


	# Set the input to 0. Don't provide any spike in input
	imgToSpikeTrain(network, np.zeros(image.shape[0]), inputIntensity)

	# Run the network without inputs to make it return to a rest state
	b2.run(restTime)

	return inputIntensity, currentIndex, accuracies







	

def trainTestSingleImage(networkList, network, trainDuration, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeImage, startTimeTraining, accuracies, 
		labelsArray, assignments, inputIntensity, startInputIntensity, 
		currentIndex, mode):


	"""
	Perform a train/test pass over an image.

	INPUT:

		1) networkList: list. Contains the network structure. Each
		element contain the number of neurons of the corresponding
		layer.

		2) network: Brian 2 Network object.

		3) trainDuration: temporal window associated to an input image

		4) spikesEvolution: numpy array. Monitor the temporal evolution
		of the spikes.

		5) updateInterval: integer. Number of images after which the
		network parameters are updated and the accuracy is evaluated.

		6) printInterval: integer. Number of images after which the
		accuracy evolution is printed

		7) currentSpikesCount: integer. Number of output spikes
		generated at the previous training pass.

		8) prevSpikesCount: integer. Total number of spikes generated
		since the beginning of the training.

		9) startTimeImage: float. Training/test on the single image
		initial time instant.

		10) startTimeTraining: float. Training/test initial time
		instant.

		11) accuracies: list. Evolution of the accuracy during the
		training/test.

		12) labelsArray: numpy array. Contains all the labels in the
		dataset.

		13) assignments: numpy array. Contains the labels currently
		associated to the output neurons.

		14) inputIntensity: float. Current value of the input scaling
		factor.

		15) startInputIntensity: float. Initial value of the input
		scaling factor.

		16) currentIndex: integer. Current image index.

		17) mode: string. Can be "train" or "test"

	OUTPUT:

		1) inputIntensity: float. Updated value of the input scaling
		factor.

		2) currentIndex: integer. Updated image index.

		3) accuracies: list. Updated evolution of the accuracy during
		the training/test.
		
	"""

	# Run the network for the desired amount of time
	network.run(trainDuration)

	# Compute the number of spikes generate by the output layer
	updatePulsesCount(network, currentSpikesCount, prevSpikesCount)

	# If the network has not been sufficiently active
	if np.sum(currentSpikesCount) < 5:

		# Repeat the training/test pass on the same image
		inputIntensity = repeatImage(inputIntensity, currentIndex)

	else:

		# Set-up to analyze the next image
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

	"""
	Set-up to analyze the next image

	INPUT:

		1) networkList: list. Contains the network structure. Each
		element contain the number of neurons of the corresponding
		layer.

		2) spikesEvolution: numpy array. Monitor the temporal evolution
		of the spikes.

		3) updateInterval: integer. Number of images after which the
		network parameters are updated and the accuracy is evaluated.

		4) printInterval: integer. Number of images after which the
		accuracy evolution is printed

		5) currentSpikesCount: integer. Number of output spikes
		generated at the previous training pass.

		6) startTimeImage: float. Training/test on the single image
		initial time instant.

		7) startTimeTraining: float. Training/test initial time
		instant.

		8) accuracies: list. Evolution of the accuracy during the
		training/test.

		9) labelsArray: numpy array. Contains all the labels in the
		dataset.

		10) assignments: numpy array. Contains the labels currently
		associated to the output neurons.

		11) startInputIntensity: float. Initial value of the input
		scaling factor.

		12) currentIndex: integer. Current image index.

		13) mode: string. Can be "train" or "test"

	OUTPUT:

		1) startInputIntensity: float. Reset value of the input scaling
		factor.

		2) currentIndex: integer. Updated image index.

		3) accuracies: list. Updated evolution of the accuracy during
		the training/test.
		
	"""

	# Add the number of spikes generated as an output to its temporal
	# evolution
	spikesEvolution[currentIndex % updateInterval] = currentSpikesCount

	# Print the training/test progress at fixed intervals
	printProgress(currentIndex, printInterval, startTimeImage, 
			startTimeTraining)

	# Update the evolution of the accuracy at fixed intervals
	accuracies = computePerformances(currentIndex, updateInterval,
			networkList[-1], spikesEvolution,
			labelsArray[currentIndex - updateInterval :
			currentIndex], assignments, accuracies)

	if mode == "train":

		# Update the labels associated to the output neurons
		updateAssignements(currentIndex, updateInterval, 
				networkList[-1], spikesEvolution, 
				labelsArray[currentIndex - updateInterval
				: currentIndex], assignments)

	return startInputIntensity, currentIndex + 1, accuracies	




def repeatImage(inputIntensity, currentIndex):

	"""
	Repeat the training/test pass on the same image.

	INPUT:
		1) inputIntensity: float. Current value of the input scaling
		factor.

		2) currentIndex: integer. Current image index.

	OUTPUT:
		Updated input intensity.
	"""

	# Print a warning to indicate that the training/test will be repeated
	print("Increase inputIntensity from " + str(inputIntensity) + \
	" to " + str(inputIntensity + 1) + " for image " + str(currentIndex))

	# Update the input intensity
	return inputIntensity + 1





def updatePulsesCount(network, currentSpikesCount, prevSpikesCount):

	"""
	Compute the number of spikes generate by the output layer.


	INPUT:
		1) network: Brian 2 Network object.

		2) currentSpikesCount: integer. Number of output spikes
		generated at the previous training pass.

		3) prevSpikesCount: integer. Total number of spikes generated
		since the beginning of the training.

	"""

	# Store the total count of the spikes for the current image
	spikeMonitorCount = network.get_states(units=True, format='dict', 
				subexpressions=False, read_only_variables=True,
				level=0)["spikemonitor"]["count"]

	# Compute the number of spikes generated only during the current image pass
	currentSpikesCount[:] = np.asarray(spikeMonitorCount) - prevSpikesCount

	# Update the total count of spikes
	prevSpikesCount[:] = np.asarray(spikeMonitorCount)




def updateAssignements(currentIndex, updateInterval, outputLayerSize,
			spikesEvolution, labelsSequence, assignments):

	"""
	Update the labels associated to the output neurons.

	INPUT:

		1) currentIndex: integer. Current image index.

		2) updateInterval: integer. Number of images after which the
		network parameters are updated and the accuracy is evaluated.

		3) outputLayerSize: integer. Number of neurons in the output
		layer.

		4) spikesEvolution: temporal evolution of the spike counter of
		each neuron for a fixed number of images.

		5) labelsSequence: numpy array. Sequence of the input labels for
		a fixed number of images.

		6) assignments: numpy array. Contains the labels currently
		associated to the output neurons.
	"""

	# Initialize maximum number of spikes generated by output neurons
	maxCount = np.zeros(outputLayerSize)

	# Update every updateInterval (fixed number of images)
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

	"""
	Print the training/test progress at fixed intervals.

	INPUT:

		1) currentIndex: integer. Current image index.

		2) printInterval: integer. Number of images after which the
		accuracy evolution is printed

		3) startTimeImage: float. Training/test on the single image
		initial time instant.

		4) startTimeTraining: float. Training/test initial time
		instant.

	"""

	# Measure the current time
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

	"""
	Update the evolution of the accuracy at fixed intervals.
	
	INPUT:

		1) currentIndex: integer. Current image index.

		2) updateInterval: integer. Number of images after which the
		network parameters are updated and the accuracy is evaluated.

		3) outputLayerSize: integer. Number of neurons in the output
		layer.

		4) spikesEvolution: temporal evolution of the spike counter of
		each neuron for a fixed number of images.

		5) labelsSequence: numpy array. Sequence of the input labels for
		a fixed number of images.

		6) assignments: numpy array. Contains the labels currently
		associated to the output neurons.

		7) accuracies: list. Evolution of the accuracy during the
		training/test.

	OUTPUT:

		accuracies: list. Updated evolution of the accuracy during
		the training/test.
	"""

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

		# Compute the accuracy and add to the temporal evolution
		accuracies = updateAccuracy(classification, labelsSequence, accuracies)

	return accuracies






def updateAccuracy(classification, labelsSequence, accuracies):

	"""
	INPUT:

		1) classification: integer. Label corresponding to the
		classification performed by the network.

		2) labelsSequence: numpy array. Sequence of the input labels for
		a fixed number of images.

		3) assignments: numpy array. Contains the labels currently
		associated to the output neurons.

	OUTPUT:

		accuracies: list. Updated evolution of the accuracy during
		the training/test.
	"""


	# Compute the number of correct classifications
	correct = np.where(classification == labelsSequence)[0].size

	# Compute the percentage of accuracy and add it to the list
	accuracies += ["{:.2f}".format(correct/classification.size*100) + "%"]
	
	# Print the accuracy
	accuracyString = "\nAccuracy: " + str(accuracies) + "\n"

	print(accuracyString)

	return accuracies
