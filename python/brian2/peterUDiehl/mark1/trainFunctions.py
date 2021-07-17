import brian2 as b2
import timeit
import numpy as np
from equationsParameters import *
from neuronsParameters import *

locals().update(parametersDict)



def trainCycle(image, networkList, network, trainDuration, restTime, 
		spikesEvolution, updateInterval, printInterval, 
		currentSpikesCount, prevSpikesCount, startTimeTraining, 
		accuracies, labelsArray, assignements, inputIntensity, 
		startInputIntensity, currentIndex):

	
	imgToSpikeTrain(network, image, inputIntensity)
	
	startTimeImage = timeit.default_timer()

	inputIntensity, currentIndex, accuracies = trainSingleImage(networkList,
		network, trainDuration, spikesEvolution, updateInterval, 
		printInterval, currentSpikesCount, prevSpikesCount, 
		startTimeImage, startTimeTraining, accuracies, labelsArray, 
		assignements, inputIntensity, startInputIntensity, currentIndex)


	imgToSpikeTrain(network, np.zeros(image.shape[0]), inputIntensity)

	b2.run(restTime)

	return inputIntensity, currentIndex, accuracies





def imgToSpikeTrain(network, image, inputIntensity):
	
	values = {
		"poissongroup":{
			"rates": image*b2.Hz/8*inputIntensity
		}
	}

	network.set_states(values, units=True, format='dict', level=0)


	

def trainSingleImage(networkList, network, trainDuration, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeImage, startTimeTraining, accuracies, 
		labelsArray, assignements, inputIntensity, startInputIntensity,
		currentIndex):

	network.run(trainDuration)

	updatePulsesCount(network, currentSpikesCount, prevSpikesCount)


	if np.sum(currentSpikesCount) < 5:

		inputIntensity = repeatImage(inputIntensity)

	else:

		inputIntensity, currentIndex, accuracies = nextImage(
			networkList, spikesEvolution, updateInterval, 
			printInterval, currentSpikesCount, startTimeImage, 
			startTimeTraining, accuracies, labelsArray, 
			assignements, startInputIntensity, currentIndex)

	return inputIntensity, currentIndex, accuracies





def nextImage(networkList, spikesEvolution, updateInterval, printInterval, 
		currentSpikesCount, startTimeImage, startTimeTraining,
		accuracies, labelsArray, assignements, startInputIntensity, 
		currentIndex):

	spikesEvolution[currentIndex % updateInterval] = currentSpikesCount

	printProgress(currentIndex, printInterval, startTimeImage,
			startTimeTraining)

	accuracies = computePerformances(currentIndex, updateInterval,
			networkList[-1], spikesEvolution,
			labelsArray[currentIndex - updateInterval :
			currentIndex], assignements, accuracies)

	updateAssignements(currentIndex, updateInterval, networkList[-1],
			spikesEvolution, labelsArray[currentIndex -
			updateInterval : currentIndex], assignements)

	return startInputIntensity, currentIndex + 1, accuracies	




def repeatImage(inputIntensity):

	print("Increase inputIntensity from " + str(inputIntensity) + \
	" to " + str(inputIntensity + 1))

	return inputIntensity + 1





def updatePulsesCount(network, currentSpikesCount, prevSpikesCount):

	# Store the count of the spikes for the current image
	spikeMonitorCount = network.get_states(units=True, format='dict', 
				subexpressions=False, read_only_variables=True,
				level=0)["spikemonitor"]["count"]

	currentSpikesCount[:] = np.asarray(spikeMonitorCount) - prevSpikesCount
	prevSpikesCount[:] = np.asarray(spikeMonitorCount)







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
