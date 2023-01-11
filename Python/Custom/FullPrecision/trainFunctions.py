import timeit
import numpy as np

from utils import seconds2hhmmss
from poisson import imgToSpikeTrain
from network import *
from common import *



def singleImageTraining(trainDuration, restTime, dt, image, network,
			networkList, dt_tauDict, stdpDict, countThreshold,
			inputIntensity, currentIndex, spikesEvolution,
			updateInterval, printInterval, startTimeTraining,
			accuracies, labelsArray, assignments,
			startInputIntensity, mode, constSums, rng):

	"""
	Train the network over an image of the dataset.

	INPUT:

		1) trainDuration: time duration of the spikes train expressed in
		milleseconds.

		2) restTime: time duration of the resting period expressed in
		milliseconds.

		3) dt: time step duration, expressed in milliseconds. 

		4) image: NumPy array containing the value of each pixel
		expressed as an integer.

		5) network: dictionary of the network.

		6) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		7) dt_tauDict: dictionary containing the exponential constants
		of the excitatory and inhibitory membrane and of the 
		homeostasis parameter theta .

		8) stdpDict: dictionary containing the STDP parameters.

		9) countThreshold: minimum acceptable number of output spikes
		generated during the training.

		10) inputIntensity: current value of the pixel"s intensity.

		11) currentIndex: index of the current image.

		12) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		13) updateInterval: number of images after which the performance
		is computed.

		14) printInterval: number of images after which the progress
		message is printed. 

		15) startTimeTraining: system time corresponfing to the beginning
		of the training.

		16) accuracies: list of strings containing the history of the
		accuracy.

		17) labelsArray: NumPy array containing all the labels of the
		training set.

		18) assignments: NumPy array containing one label assignment for
		each output neuron.

		19) startInputIntensity: starting value of the pixel"s intensity.
		The default value is 2.

		20) mode: string. It can be "train" or "test".

		21) constSums: NumPy array. Each element represents the constant
		value corresponding to the sum of all the weights of a single 
		neuron in the specific layer.

		22) rng: NumPy random generator.

	
	OUTPUT:

		1) inputIntensity: update value of the pixel"s intensity.

		2) currentIndex: index of the next image to analyse.

		3) accuracies: updated list of strings containing the history of 
		the accuracy.

	"""


	# Convert the time into number of training steps
	trainingSteps = int(trainDuration/dt)
	restingSteps = int(restTime/dt)

	# Measure the system time corresponding to the beginning of the image
	startTimeImage = timeit.default_timer()

	# Convert the image into spikes trains
	spikesTrains = imgToSpikeTrain(image, dt, trainingSteps, inputIntensity,
			rng)

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
			mode,
			constSums
			)


	# Bring the network into a rest state
	rest(network, networkList)


	return inputIntensity, currentIndex, accuracies






def train(network, networkList, spikesTrains, dt_tauDict, stdpDict,
		countThreshold, inputIntensity, currentIndex, spikesEvolution,
		updateInterval, printInterval, startTimeImage,
		startTimeTraining, accuracies, labelsArray, assignments,
		startInputIntensity, mode, constSums):

	"""
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

		7) inputIntensity: current value of the pixel"s intensity.

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

		17) startInputIntensity: starting value of the pixel"s intensity.
		The default value is 2.

		18) mode: string. It can be "train" or "test".

		19) constSums: numpy array. Normalization factor of the weights
		for each layer.

	OUTPUT:

		1) inputIntensity: update value of the pixel"s intensity.

		2) currentIndex: index of the next image to analyse.

		3) accuracies: updated list of strings containing the history of 
		the accuracy.


	"""


	
	# Train the network over the pixels" spikes train
	spikesCounter = run(network, networkList, spikesTrains, dt_tauDict,
				stdpDict, mode, constSums)

	if np.sum(spikesCounter) < countThreshold:

		# Prepare the training over the same image
		inputIntensity = repeatImage(inputIntensity, currentIndex)

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




def rest(network, networkList):

	"""
	Bring the network into a rest state.

	INPUT:

		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.
	"""

	for layer in range(1, len(networkList)):

		# Reset the membrane potential to the rest value
		network["excLayer" + str(layer)]["v"][0][:] = network["excLayer"
			+ str(layer)]["vRest"]
