from brian2 import run
import timeit
import numpy as np



def updateAssignements(currentIndex, updateInterval, outputLayerSize,
			spikesEvolution, labelsSequence, assignements):

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

			# Update the assignements	
			assignements[whereMaxSpikes] = label

			# Update the maxima
			maxCount[whereMaxSpikes] = labelSpikes[whereMaxSpikes]




def printProgress(currentIndex, printInterval, startTimeImage, startTimeTraining):

	# Print every printInterval
	if currentIndex % printInterval == 0 and currentIndex > 0:
			
		progressString = "Analyzed images: " + str(currentIndex) + \
		". Time required for a single image: " + \
		str(timeit.default_timer() - startTimeImage) + \
		"s. Total elapsed time: " + \
		str(timeit.default_timer() - startTimeTraining) + "s."

		print(progressString)



def computePerformances(currentIndex, updateInterval, outputLayerSize,
			spikesEvolution, labelsSequence, assignements, accuracies):

	maxCount = np.zeros(updateInterval)
	classification = -1*np.ones(updateInterval)

	# Update every updateInterval
	if currentIndex % updateInterval == 0 and currentIndex > 0:

		# Loop over the ten labels
		for label in range(10):

			# Consider only the neurons assigned to the current
			# label
			spikeCount = np.sum(spikesEvolution[:, assignements ==
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
	print("{:.2f}".format(correct/classification.size*100) + "%")
	accuracies += ["{:.2f}".format(correct/classification.size*100) + "%"]
	
	# Print the accuracy
	accuracyString = "\nAccuracy: " + str(accuracies) + "\n"

	print(accuracyString)

	return accuracies
