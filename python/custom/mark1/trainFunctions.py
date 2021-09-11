import timeit
import numpy as np
import sys

from utils import seconds2hhmmss
from poisson import imgToSpikeTrain
from network import run



def singleImageTraining(trainDuration, restTime, dt, image, pixelMin, pixelMax,
			network, networkList, dt_tauDict, stdpDict,
			countThreshold, inputIntensity, currentIndex,
			spikesEvolution, updateInterval, printInterval,
			startTimeTraining, accuracies, labelsArray, assignments,
			startInputIntensity, mode, constSums):

	'''
	Train the network over an image of the dataset.

	INPUT:

		1) trainDuration: time duration of the spikes train expressed in
		milleseconds.

		2) restTime: time duration of the resting period expressed in
		milliseconds.

		3) dt: time step duration, expressed in milliseconds. 

		4) image: NumPy array containing the value of each pixel
		expressed as an integer.

		5) pixelMin: minimum value of the pixels. In the MNIST the value
		0 is used to express a totally black pixel.

		6) pixelMax: maximum value of the pixels. In the MNIST the value
		255 is used to express a totally white pixel.

		7) network: dictionary of the network.

		8) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		9) dt_tauDict: dictionary containing the exponential constants
		of the excitatory and inhibitory membrane and of the 
		homeostasis parameter theta .

		10) stdpDict: dictionary containing the STDP parameters.

		11) countThreshold: minimum acceptable number of output spikes
		generated during the training.

		12) inputIntensity: current value of the pixel's intensity.

		13) currentIndex: index of the current image.

		14) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		15) updateInterval: number of images after which the performance
		is computed.

		16) printInterval: number of images after which the progress
		message is printed. 

		17) startTimeTraining: system time corresponfing to the beginning
		of the training.

		18) accuracies: list of strings containing the history of the
		accuracy.

		19) labelsArray: NumPy array containing all the labels of the
		training set.

		20) assignments: NumPy array containing one label assignment for
		each output neuron.

		21) startInputIntensity: starting value of the pixel's intensity.
		The default value is 2.

		22) mode: string. It can be "train" or "test".

		23) constSums: NumPy array. Each element represents the constant
		value corresponding to the sum of all the weights of a single 
		neuron in the specific layer.

	
	OUTPUT:

		1) inputIntensity: update value of the pixel's intensity.

		2) currentIndex: index of the next image to analyse.

		3) accuracies: updated list of strings containing the history of 
		the accuracy.

	'''


	

	# Convert the time into number of training steps
	trainingSteps = int(trainDuration/dt)
	restingSteps = int(restTime/dt)

	# Measure the system time corresponding to the beginning of the image
	startTimeImage = timeit.default_timer()

	# Convert the image into spikes trains
	spikesTrains = imgToSpikeTrain(image, trainingSteps, pixelMin, pixelMax,
			inputIntensity)

	# Train the network with the spikes sequences associated to the pixels.
	inputIntensity, currentIndex, accuracies = \
		train(
			network, 
			networkList, 
			spikesTrains, 
			dt_tauDict, 
			stdpDict,
			countThreshold,
			inputIntensity, 
			currentIndex, 
			spikesEvolution, 
			updateInterval,
			printInterval, 
			startTimeImage, 
			startTimeTraining, 
			accuracies, 
			labelsArray, 
			assignments, 
			startInputIntensity, 
			mode
			)

	# Normalize the weights
	normalizeWeights(network, networkList, constSums)

	# Bring the network into a rest state
	rest(network, networkList, restingSteps, image.shape[0], dt_tauDict,
		stdpDict)


	return inputIntensity, currentIndex, accuracies






def train(network, networkList, spikesTrains, dt_tauDict, stdpDict,
		countThreshold, inputIntensity, currentIndex, spikesEvolution,
		updateInterval, printInterval, startTimeImage,
		startTimeTraining, accuracies, labelsArray, assignments,
		startInputIntensity, mode):

	'''
	Train the network with the spikes sequences associated to the pixels.

	INPUT:

		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) spikesTrains: two-dimensional NumPy array. One training step
		for each row; one input for each column.

		4) dt_tauDict: dictionary containing the exponential constants
		of the excitatory and inhibitory membrane and of the 
		homeostasis parameter theta .

		5) stdpDict: dictionary containing the STDP parameters.

		6) countThreshold: minimum acceptable number of output spikes
		generated during the training.

		7) inputIntensity: current value of the pixel's intensity.

		8) currentIndex: index of the current image.

		9) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		10) updateInterval: number of images after which the performance
		is computed.

		11) printInterval: number of images after which the progress
		message is printed. 

		12) startTimeImage: system time corresponding to the beginning of
		the image.

		13) startTimeTraining: system time corresponfing to the beginning
		of the training.

		14) accuracies: list of strings containing the history of the
		accuracy.

		15) labelsArray: NumPy array containing all the labels of the
		training set.

		16) assignments: NumPy array containing one label assignment for
		each output neuron.

		17) startInputIntensity: starting value of the pixel's intensity.
		The default value is 2.

		18) mode: string. It can be "train" or "test".

	OUTPUT:

		1) inputIntensity: update value of the pixel's intensity.

		2) currentIndex: index of the next image to analyse.

		3) accuracies: updated list of strings containing the history of 
		the accuracy.


	'''


	
	# Train the network over the pixels' spikes train
	spikesCounter = run(network, networkList, spikesTrains, dt_tauDict,
				stdpDict)

	if np.sum(spikesCounter) < countThreshold:

		if inputIntensity < 10:

			# Prepare the training over the same image
			inputIntensity = repeatImage(inputIntensity, currentIndex)

		else:
			inputIntensity = startInputIntensity
			currentIndex += 1

	else:
		
		# Prepare the training over the next image
		inputIntensity, currentIndex, accuracies = \
			nextImage(
				networkList, 
				spikesEvolution, 
				updateInterval, 
				printInterval, 
				spikesCounter, 
				startTimeImage,
				startTimeTraining, 
				accuracies, 
				labelsArray, 
				assignments, 
				startInputIntensity, 
				currentIndex, 
				mode
			)

	return inputIntensity, currentIndex, accuracies






def nextImage(networkList, spikesEvolution, updateInterval, printInterval,
	spikesCounter, startTimeImage, startTimeTraining, accuracies,
	labelsArray, assignments, startInputIntensity, currentIndex, mode):


	'''
	Prepare the training over the next image.

	INPUT:

		1) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		3) updateInterval: number of images after which the performance
		is computed.

		4) printInterval: number of images after which the progress
		message is printed. 

		5) spikesCounter: NumPy array containing the total amount of 
		generate spike for each neuron.

		6) startTimeImage: system time corresponding to the beginning of
		the image.

		7) startTimeTraining: system time corresponfing to the beginning
		of the training.

		8) accuracies: list of strings containing the history of the
		accuracy.

		9) labelsArray: NumPy array containing all the labels of the
		training set.

		10) assignments: NumPy array containing one label assignment for
		each output neuron.

		11) startInputIntensity: starting value of the pixel's intensity.
		The default value is 2.

		12) currentIndex: index of the current image.

		13) mode: string. It can be "train" or "test".


	OUTPUT:

		1) startInputIntensity: reset value of the pixel's intensity.

		2) currentIndex + 1: index of the next image to analyse.

		3) accuracies: updated list of strings containing the history of 
		the accuracy.

	'''

	# Update the temporal evolution of the spikes
	spikesEvolution[currentIndex % updateInterval] = spikesCounter

	# Print the training progress
	printProgress(currentIndex, printInterval, startTimeImage,
		startTimeTraining)

	# Compute the accuracy of the network
	accuracies = computePerformance(
				currentIndex, 
				updateInterval,
				networkList[-1], 
				spikesEvolution, 
				labelsArray[currentIndex - updateInterval :
					currentIndex], 
				assignments, 
				accuracies
			)


	if mode == "train": 

		# Update the output classification
		updateAssignments(
				currentIndex, 
				updateInterval,
				networkList[-1], 
				spikesEvolution, 
				labelsArray[currentIndex - updateInterval :
					currentIndex], 
				assignments
		)

	# Reset input intensity and increase the image index by 1
	return startInputIntensity, currentIndex + 1, accuracies	







def printProgress(currentIndex, printInterval, startTimeImage, 
			startTimeTraining):

	'''
	Print the training progress.

	INPUT:

		1) currentIndex: index of the current image.

		2) printInterval: number of images after which the progress
		message is printed. 

		3) startTimeImage: system time corresponding to the beginning of
		the image.

		4) startTimeTraining: system time corresponfing to the beginning
		of the training.

	'''

	# End of print interval?
	if currentIndex % printInterval == 0 and currentIndex > 0:

		# Measure the current time instant
		currentTime = timeit.default_timer()
			
		# Format the output message and print it
		progressString = "Analyzed images: " + str(currentIndex) + \
			". Time required for a single image: " + str(currentTime
			- startTimeImage) + "s. Total elapsed time: " + \
			seconds2hhmmss(timeit.default_timer() - 
			startTimeTraining)

		print(progressString)




def computePerformance(currentIndex, updateInterval, lastLayerSize,
			spikesEvolution, labelsSequence, assignments, 
			accuracies):

	'''
	Compute the network performance.

	INPUT:	

		1) currentIndex: index of the current image.

		2) updateInterval: number of images after which the performance
		is computed.

		3) lastLayerSize: number of elements in the output layer.

		4) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		5) labelsSequence: NumPy array containing the history of the
		labels in the last "updateInterval" cycles.

		6) assignments: NumPy array containing one label assignment for
		each output neuron.

		7) accuracies: list of strings containing the history of the
		accuracy.

	OUTPUT:

		accuracies: updated list of strings containing the history of the
		accuracy.

	'''


	# End of update interval?
	if currentIndex % updateInterval == 0 and currentIndex > 0:

		# Initialize the maximum count to 0
		maxCount = np.zeros(updateInterval)

		# Initialize the output classification
		classification = -1*np.ones(updateInterval)


		for label in range(10):

			# Add the spikes count associated to the current label
			spikesCount = np.sum(spikesEvolution[:, assignments ==
					label], axis = 1)

			# Find where the spikes count is grater than the maximum
			whereMaxSpikes = spikesCount > maxCount

			# Associate the instants to the current label
			classification[whereMaxSpikes] = label

			# Update the maximum number of spikes for the label
			maxCount[whereMaxSpikes] = spikesCount[whereMaxSpikes]

		# Compute the accuracy and add it to the list of accuracies
		accuracies = updateAccuracy(classification, labelsSequence, accuracies)

	return accuracies







def updateAccuracy(classification, labelsSequence, accuracies):

	'''
	Compute the accuracy and add it to the list of accuracies.

	INPUT:

		1) classification: NumPy array containing the history of the
		classification performed by the network in the last
		"updateInterval" cycles

		2) labelsSequence: NumPy array containing the history of the
		labels in the last "updateInterval" cycles.

		3) accuracies: list of strings containing the history of the
		accuracy.

	OUTPUT:

		accuracies: updated list of strings containing the history of the
		accuracy.

	'''

	# Number of instants in which the classification is equal to the label
	correct = np.where(classification == labelsSequence)[0].size

	# Compute the percentage of accuracy and add it to the list
	accuracies += ["{:.2f}".format(correct/classification.size*100) + "%"]
	
	# Print the accuracy
	accuracyString = "\nAccuracy: " + str(accuracies) + "\n"

	print(accuracyString)

	return accuracies






def updateAssignments(currentIndex, updateInterval, lastLayerSize,
			spikesEvolution, labelsSequence, assignments):

	'''
	Update the output classification.

	INPUT:

		1) currentIndex: index of the current image.

		2) updateInterval: number of images after which the performance
		is computed.

		3) lastLayerSize: number of elements in the output layer.

		4) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		5) labelsSequence: NumPy array containing the history of the
		labels in the last "updateInterval" cycles.

		6) assignments: NumPy array containing one label assignment for
		each output neuron.

	'''

	# End of update interval?
	if currentIndex % updateInterval == 0 and currentIndex > 0:

		# Initialize the maximum count to 0
		maxCount = np.zeros(lastLayerSize)

		for label in range(10):

			# Add spikes for the instants associated to the label
			labelSpikes = np.sum(spikesEvolution[labelsSequence ==
					label], axis=0)

			# Find where spikes count exceeds current maximum
			whereMaxSpikes = labelSpikes > maxCount

			# Update the assignments	
			assignments[whereMaxSpikes] = label

			# Update the maximum count for the current label
			maxCount[whereMaxSpikes] = labelSpikes[whereMaxSpikes]







def repeatImage(inputIntensity, currentIndex):

	'''
	Prepare the training over the same image.

	INPUT:

		1) inputIntensity: current value of the pixel's intensity.

		2) currentIndex: index of the current image.

	OUTPUT:

		current value of the pixel's intensity increased by 1.

	'''

	# Print a message to say that the training will be repeated
	print("Increase inputIntensity from " + str(inputIntensity) + \
	" to " + str(inputIntensity + 1) + " for image " + str(currentIndex))

	# Increase the pixel's intensity
	return inputIntensity + 1






def normalizeWeights(network, networkList, constSums):

	'''
	Normalize the weights of all the layers in the network.

	INPUT:

		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) constSums: NumPy array. Each element represents the constant
		value corresponding to the sum of all the weights of a single 
		neuron in the specific layer.
		
	'''
	
	for layer in range(1, len(networkList)):

		# Normalize the weights of the layer
		normalizeLayerWeights(network, "exc2exc" + str(layer),
					constSums[layer - 1])






def normalizeLayerWeights(network, synapseName, constSum):

	'''
	Normalize the weights of the given layer.

	INPUT:

		1) network: dictionary of the network.


		2) synapseName: string reporting the name of the connection. The
		standard name is "exc2exc" + the index of the layer.

		3) constSum: constant value corresponding to the sum of all the
		weights of a single neuron.

	'''

	# Compute the sum of the weights for each neuron
	weightsSum = np.sum(network[synapseName]["weights"], 
			axis = 1).reshape(network[synapseName]\
			["weights"].shape[0], 1)

	# Set to one the zero sums to avoid division by 0
	weightsSum[weightsSum == 0] = 1	

	# Compute the normalization factor
	normFactor = constSum / weightsSum

	# Normalize the weights
	network[synapseName]["weights"][:] *= normFactor



def rest(network, networkList, restingSteps, imageSize, dt_tauDict, stdpDict):

	'''
	Bring the network into a rest state.

	INPUT:

		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) restingSteps: time duration of the resting period expressed
		in trainingSteps.

		4) imageSize: total number of pixels composing the image.

		5) dt_tauDict: dictionary containing the exponential constants
		of the excitatory and inhibitory membrane and of the 
		homeostasis parameter theta .

		6) stdpDict: dictionary containing the STDP parameters.

	'''

	# Reset to zero the spikes trains
	spikesTrains = np.zeros((restingSteps, imageSize)).astype(bool)

	# Run the network on the resting inputs
	run(network, networkList, spikesTrains, dt_tauDict, stdpDict)

	# Reset the time instants
	network["exc2exc1"]["t_in"][:] = 0
	network["exc2exc1"]["t_out"][:] = 0

	# Reset the masks
	network["exc2exc1"]["mask_in"][:] = False
	network["exc2exc1"]["mask_out"][:] = False
