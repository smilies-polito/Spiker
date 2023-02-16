import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit

from files import *

development = "../"

if development not in sys.path:
	sys.path.insert(1,development)

print(sys.path)


if mnistDir not in sys.path:
	sys.path.append(mnistDir)

from mnist import loadDataset
from createNetwork import createNetwork
from poisson import imgToSpikeTrain
from network import run
from storeParameters import storeArray

# Initialize the training parameters
from runParameters import *

# storeArray(assignmentsFile, -1*np.ones(networkList[-1]))
# storeArray(weightFilename + "1.npy", np.random.rand(networkList[1],
# networkList[0])
# + 0.01)
# storeArray(thresholdFilename + "1.npy", -1*np.ones(networkList[-1]))

# Load the MNIST dataset
imgArray, labelsArray = loadDataset(trainImages, trainLabels)



# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights)

# Convert the train duration from milliseconds to elaboration steps
trainDuration = int(trainDuration/dt)

finalLabel = -1


for i in range(100):

	# Choose an image to give in input to the network
	image = imgArray[i]


	# Plot the image in order to visualize it
	plt.imshow(image.reshape(28,28), cmap = 'gray')
	plt.show()


	# Convert the image into trains of spikes
	spikesTrains = imgToSpikeTrain(image, dt, trainDuration, inputIntensity,
			rng)

	spikesCounter = np.zeros(networkList[-1])

	while np.sum(spikesCounter) < 5:
		# Run a test of the network on the choosen image
		start = timeit.default_timer()

		spikesCounter = run(network, networkList, spikesTrains, dt_tauDict, stdpDict, 
				mode, constSums)

		stop = timeit.default_timer()
		print("Total time: ", stop - start)


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
