import numpy as np
import timeit

from utils import seconds2hhmmss
from network import run 


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






def nextImage(networkList, spikesEvolution, updateInterval, printInterval,
	spikesCounter, startTimeImage, startTimeTraining, accuracies,
	labelsArray, assignments, startInputIntensity, currentIndex, labels, mode):


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

		13) labels: NumPy array containing the whole list of labels.

		14) mode: string. It can be "train" or "test".


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
				assignments,
				labels
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




def computePerformance(currentIndex, updateInterval, spikesEvolution, 
			labelsSequence, assignments, accuracies):

	'''
	Compute the network performance.

	INPUT:	

		1) currentIndex: index of the current image.

		2) updateInterval: number of images after which the performance
		is computed.

		3) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		4) labelsSequence: NumPy array containing the history of the
		labels in the last "updateInterval" cycles.

		5) assignments: NumPy array containing one label assignment for
		each output neuron.

		6) accuracies: list of strings containing the history of the
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
			spikesEvolution, labelsSequence, assignments, labels):

	'''
	Update the output classification. The number of neurons and the number
	of labels must be the same (lastLayerSize = labels.shape[0]).

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

		7) labels: NumPy array containing the whole list of labels.


	'''

	# End of update interval?
	if currentIndex % updateInterval == 0 and currentIndex > 0:

		neuronsAssigned = np.zeros(lastLayerSize).astype(bool)
		labelsAssigned = np.zeros(labels.shape[0]).astype(bool)

		# Assign neurons that have generated a maximum count
		assignMaxCounts(spikesEvolution, neuronsAssigned,
				labelsAssigned, assignments, labels,
				labelsSequence)

		# Assign remaining neurons
		assignNoMaxCounts(assignments, neuronsAssigned, labels,
				labelsAssigned)



def assignMaxCounts(spikesEvolution, neuronsAssigned, labelsAssigned,
		assignments, labels, labelsSequence):

	'''
	Search the most active neuron for each label and assign it to the label.

	INPUT:

		1) spikesEvolution: two-dimensional NumPy array containing the
		history of the spikes counter in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		
		2) neuronsAssigned: NumPy array of boolean values. One value for
		each neuron.

		3) labelsAssigned: NumPy array of boolean values. One value for
		each label.

		4) assignments: NumPy array containing the assigned labels. One
		label for each neuron.

		5) labels: NumPy array containing the whole list of labels.

		6) labelsSequence: NumPy array containing the history of the
		labels in the last "updateInterval" cycles.

	'''

	# Loop over all the labels
	for label in labels:

		# Compute the total counts of spikes for the current label
		spikesCounters = np.sum(spikesEvolution[labelsSequence ==
			label], axis = 0)

		# Search and assign the most active neuron for the current label
		searchMaxAndUppdate(spikesCounters, label, neuronsAssigned,
			labelsAssigned, assignments)




def searchMaxAndUppdate(spikesCounters, label, neuronsAssigned, labelsAssigned,
		assignments):

	'''
	Search the most active neuron for a specific label and assign it to the
	label.

	INPUT:
		
		1) spikesCounters: NumPy array containing the total amount of 
		spikes generated by each neuron for the current label.

		2) label: current label.

		3) neuronsAssigned: NumPy array of boolean values. One value for
		each neuron.

		4) labelsAssigned: NumPy array of boolean values. One value for
		each label.

		5) assignments: NumPy array containing the assigned labels. One
		label for each neuron.

	'''

	maxCount = 0
	maxFound = 0

	# Loop over all the output neurons
	for i in range(spikesCounters.shape[0]):

		# If spikes count for specific neuron exceeds maximum
		if spikesCounters[i] > maxCount and neuronsAssigned[i] == False:

			maxFound = 1
			maxIndex = i
			maxCount = spikesCounters[i]

	if maxFound ==1:

		# Assign the current label to the neuron
		assignments[maxIndex] = label

		# Remove the neuron from the next search
		neuronsAssigned[maxIndex] = True

		# Remove the label from the next search
		labelsAssigned[label] = True



def assignNoMaxCounts(assignments, neuronsAssigned, labels, labelsAssigned):

	'''
	Assign the labels for which no neurons have generated a maximum count.
	The function requires that the number of neurons and labels is the same,
	one neuron for each label.

	INPUT:

		1) assignments: NumPy array containing the assigned labels. One
		label for each neuron.

		2) neuronsAssigned: NumPy array of boolean values. One value for
		each neuron.

		3) labels: NumPy array containing the whole list of labels.

		4) labelsAssigned: NumPy array of boolean values. One value for
		each label.

	'''

	# Assign remaining labels to remaining neruons
	assignments[np.where(neuronsAssigned == False)] =\
	labels[np.where(labelsAssigned == False)]
