import numpy as np
import matplotlib.pyplot as plt
import sys

development = "../"

if development not in sys.path:
	sys.path.insert(1,development)



from mnist import loadDataset
from createNetwork import createNetwork
from poisson import imgToSpikeTrain
from network import run

# Initialize the training parameters
from files import *
from runParameters import *



# Load the MNIST dataset
imgArray, labelsArray = loadDataset(trainImages, trainLabels)


# Load the MNIST dataset
imgArray, labelsArray = loadDataset(trainImages, trainLabels)




# Create the network data structure
network = createNetwork(networkList, weightFilename, thetaFilename, mode, 
			excDictList, inhDictList, scaleFactors, exc2inhWeights,
			inh2excWeights)

# Convert the train duration from milliseconds to elaboration steps
trainDuration = int(trainDuration/dt)


for i in range(100):

	# Choose an image to give in input to the network
	image = imgArray[i]


	# Plot the image in order to visualize it
	plt.imshow(image.reshape(28,28), cmap = 'gray')
	plt.show()


	# Convert the image into trains of spikes
	spikesTrains = imgToSpikeTrain(image, dt, trainDuration, inputIntensity)

	spikesCounter = np.zeros(networkList[-1])

	while np.sum(spikesCounter) < 5:
		# Run a test of the network on the choosen image
		spikesCounter = run(network, networkList, spikesTrains, dt_tauDict, stdpDict, 
				mode)

	# Find the labels associated to the neurons that have generated the maximum
	# number of spikes
	labels = assignments[spikesCounter[0] == np.max(spikesCounter)]

	# Find the label with the maximum amount of occurences
	maxCount = 0
	for label in range(10):
		if np.sum(labels == label) > maxCount:
			max = np.sum(labels == label)
			finalLabel = label


	print("\nThe number represented in the image is: ", finalLabel)
	print("\n")
