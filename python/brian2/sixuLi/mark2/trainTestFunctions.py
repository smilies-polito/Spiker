import brian2 as b2
import timeit
import numpy as np
from equationsParameters import *
from neuronsParameters import *
import sys

locals().update(parametersDict)


def initAssignements(mode, networkList, assignementsFile):

	if mode == "train":
		return -1*np.ones(networkList[-1])

	elif mode == "test":
		with open(assignementsFile, 'rb') as fp:
			return np.load(fp)

	else:
		print('Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train')
		sys.exit()



def trainTestCycle(image, networkList, network, trainDuration, restTime, 
		spikesEvolution, updateInterval, printInterval, 
		currentSpikesCount, prevSpikesCount, startTimeTraining, 
		accuracies, labelsArray, assignements, inputIntensity, 
		startInputIntensity, currentIndex, mode, constSum):

	startTimeImage = timeit.default_timer()

	imgToSpikeTrain(network, image, inputIntensity)
	
	inputIntensity, currentIndex, accuracies = trainTestSingleImage(
		networkList, network, trainDuration, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeImage, startTimeTraining, accuracies,
		labelsArray, assignements, inputIntensity, startInputIntensity, 
		currentIndex,mode)



	normalizeNetWeights(network, networkList, constSum)	

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


	

def trainTestSingleImage(networkList, network, trainDuration, spikesEvolution, 
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





def nextImage(networkList, spikesEvolution, updateInterval, printInterval, 
		currentSpikesCount, startTimeImage, startTimeTraining, 
		accuracies, labelsArray, assignements, startInputIntensity, 
		currentIndex, mode):

	spikesEvolution[currentIndex % updateInterval] = currentSpikesCount

	printProgress(currentIndex, printInterval, startTimeImage, 
			startTimeTraining)

	accuracies = computePerformances(currentIndex, updateInterval,
			networkList[-1], spikesEvolution,
			labelsArray[currentIndex - updateInterval :
			currentIndex], assignements, accuracies)

	if mode == "train":
		updateAssignements(currentIndex, updateInterval, 
				networkList[-1], spikesEvolution, 
				labelsArray[currentIndex - updateInterval
				: currentIndex], assignements)

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
	accuracies += ["{:.2f}".format(correct/classification.size*100) + "%"]
	
	# Print the accuracy
	accuracyString = "\nAccuracy: " + str(accuracies) + "\n"

	print(accuracyString)

	return accuracies





def storeParameters(networkList, network, assignements, weightFilename,
			thetaFilename, assignementsFilename):

	storeArray(weightFilename + str(1) + ".npy", network["poisson2exc"].w)

	storeArray(thetaFilename + str(1) + ".npy", network["excLayer1"].theta)

	for layer in range(2, len(networkList)):

		storeArray(weightFilename + str(layer) + ".npy", 
			network["exc2exc" + str(layer)].w)

		storeArray(thetaFilename + str(layer) + ".npy", 
			network["excLayer" + str(layer)].theta)


	storeArray(assignementsFilename + ".npy", assignements)




def storeArray(filename, numpyArray):

	with open(filename, 'wb') as fp:
		np.save(fp, numpyArray)




def storePerformace(startTimeTraining, accuracies, performanceFilename):

	timeString = "Total training time : " + \
		seconds2hhmmss(timeit.default_timer() - startTimeTraining)

	accuracyString = "Accuracy evolution:\n" + "\n".join(accuracies)

	with open(performanceFilename + ".txt", 'w') as fp:
		fp.write(timeString)
		fp.write("\n\n")
		fp.write(accuracyString)


def seconds2hhmmss(seconds):

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	seconds = int(seconds % 60)

	return str(hours) + "h " + str(minutes) + "min " + str(seconds) + "s"



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
