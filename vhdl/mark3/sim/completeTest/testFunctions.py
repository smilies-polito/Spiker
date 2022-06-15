import timeit
import numpy as np

from utils import seconds2hhmmss, storeBinaryArray_txt
from poisson import imgToSpikeTrain
from network import run
from common import *



def singleImageTest(trainDuration, restTime, dt, image, networkScript,
	countersFilename, networkList, dt_tauDict, countThreshold,
	inputIntensity, currentIndex, spikesEvolution, updateInterval,
	printInterval, startTimeTraining, accuracies, labelsArray, assignments,
	startInputIntensity, mode, constSums, rng, exp_shift,
	neuron_parallelism, spikesFilename, countBitWidth):

	'''
	Test the network over an image of the dataset.

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
		o6f the excitatory and inhibitory membrane and of the 
		homeostasis parameter theta .

		8) countThreshold: minimum acceptable number of output spikes
		generated during the training.

		9) inputIntensity: current value of the pixel's intensity.

		10) currentIndex: index of the current image.

		11) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		12) updateInterval: number of images after which the performance
		is computed.

		13) printInterval: number of images after which the progress
		message is printed. 

		14) startTimeTraining: system time corresponfing to the beginning
		of the training.

		15) accuracies: list of strings containing the history of the
		accuracy.0
		15) labelsArray: NumPy array containing all the labels of the
		training set.

		16) labelsArray: NumPy array containing all the labels of the
		training set.

		17) assignments: NumPy array containing one label assignment for
		each output neuron.

		18) startInputIntensity: starting value of the pixel's intensity.
		The default value is 2.

		19) mode: string. It can be "train" or "test".

		20) constSums: NumPy array. Each element represents the constant
		value corresponding to the sum of all the weights of a single 
		neuron in the specific layer.
		
		21) rng: NumPy random generator.
		
		22) exp_shift: bit shift for the exponential decay.

		23) neuron_parallelism: number of bits on which the neuron works.

		24) spikesFilename: string. Name of the file containing the input
		spikes.

	
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

	storeBinaryArray_txt(imageFilename, image)
	
	# Test the network with the spikes sequences associated to the pixels.
	inputIntensity, currentIndex, accuracies = \
		test(
			networkScript, 
			countersFilename,
			networkList, 
			spikesTrains, 
			dt_tauDict, 
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
			constSums,
			exp_shift,
			neuron_parallelism,
			countBitWidth
			)


	return inputIntensity, currentIndex, accuracies






def test(networkScript, countersFilename, networkList, spikesTrains, dt_tauDict,
	countThreshold, inputIntensity, currentIndex, spikesEvolution,
	updateInterval, printInterval, startTimeImage, startTimeTraining,
	accuracies, labelsArray, assignments, startInputIntensity, mode,
	constSums, exp_shift, neuron_parallelism, countBitWidth):

	'''
	Test the network with the spikes sequences associated to the pixels.

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

		5) countThreshold: minimum acceptable number of output spikes
		generated during the training.

		6) inputIntensity: current value of the pixel's intensity.

		7) currentIndex: index of the current image.

		8) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		9) updateInterval: number of images after which the performance
		is computed.

		10) printInterval: number of images after which the progress
		message is printed. 

		11) startTimeImage: system time corresponding to the beginning of
		the image.

		12) startTimeTraining: system time corresponfing to the beginning
		of the training.

		13) accuracies: list of strings containing the history of the
		accuracy.

		14) labelsArray: NumPy array containing all the labels of the
		training set.

		15) assignments: NumPy array containing one label assignment for
		each output neuron.

		16) startInputIntensity: starting value of the pixel's intensity.
		The default value is 2.

		17) mode: string. It can be "train" or "test".
		
		18) exp_shift: bit shift for the exponential decay.

		19) neuron_parallelism: number of bits on which the neuron works.

	OUTPUT:

		1) inputIntensity: update value of the pixel's intensity.

		2) currentIndex: index of the next image to analyse.

		3) accuracies: updated list of strings containing the history of 
		the accuracy.


	'''

	
	# Train the network over the pixels' spikes train
	spikesCounter = run(networkScript, countersFilename, networkList[-1],
			countBitWidth)


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
