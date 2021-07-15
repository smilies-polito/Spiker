from brian2 import run
import timeit
import numpy as np




def trainSingleImg(network, image, ratesDict, startTimeTraining):

	ratesDict["poissongroup"]["rates"] = image*b2.Hz
	network.set_states(ratesDict, units=True, format='dict', level=0 )

	network.run(200*b2.ms)

	print(network.get_states(units=True, format='dict', subexpressions=False,
	read_only_variables=True, level=0)["poissongroup"]["rates"])

	return network

def trainCycle():

	# Measure the time at which the training on the single image starts
	startTimeImage = timeit.default_timer()

	# Train the network
	b2.run(singleExampleTime)
	

def updateSimParameters(networkList, network, prevSpikesCount, inputIntensity,
			startInputIntensity, currentIndex, updateInterval, 
			printInterval, startTimeImage, startTimeTraining, 
			accuracies, spikesEvolution, labelsArray, assignements):

	currentSpikesCount, prevSpikesCount = updatePulsesCount(network, 
							prevSpikesCount)

	if np.sum(currentSpikesCount) < 5:

		inputIntensity +=1
		print("Increase input intensity: ", inputIntensity)

	else:

		spikesEvolution[currentIndex % updateInterval] = 
				currentSpikesCount

		printProgress(currentIndex, printInterval, startTimeImage, 
				startTimeTraining)

		accuracies = computePerformances(currentIndex, updateInterval,
				networkList[-1], spikesEvolution, 
				labelsArray[currentIndex - updateInterval :
				currentIndex], assignements, accuracies)

		# Update the correspondence between the output neurons and the labels
		updateAssignements(currentIndex, updateInterval, 
				networkList[1], spikesEvolution, 
				labelsArray[currentIndex - updateInterval :
				currentIndex], assignements)

		inputIntensity = startInputIntensity

		currentIndex += 1
	
	return prevSpikesCount, inputIntensity, currentIndex




def updatePulsesCount(network, currentSpikesCount, prevSpikesCount):

	# Store the count of the spikes for the current image
	spikeMonitorCount = network.get_states(units=True, format='dict', 
				subexpressions=False, read_only_variables=True,
				level=0)["spikemonitor"]["count"]

	currentSpikesCount =  spikeMonitorCount - prevSpikesCount
	prevSpikesCount = np.asarray(spikeMonitorCount)





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


from createNetwork import *
from equations import *
from equationsParameters import *
from neuronsParameters import *

locals().update(parametersDict)

networkList = [784, 5, 1]
image = np.linspace(1, 10, 10)

ratesDict = {
	"poissongroup"	: {
		"rates"		: np.ones(784)
	}
}

network = createNetwork(networkList, equationsDict, parametersDict, 
		stdpDict, weightInitDict)


image = 50000*np.ones(784)
startTimeTraining = 1
network = trainSingleImg(network, image, ratesDict, startTimeTraining)

prevSpikesCount = 0

updatePulsesCount(network, prevSpikesCount)
