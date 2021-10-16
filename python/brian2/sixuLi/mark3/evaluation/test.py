import numpy as np
import brian2 as b2
import timeit
import subprocess

import sys

development = "../"

if development not in sys.path:
	sys.path.insert(1,development)


from mnist import loadDataset
from createNetwork import createNetwork
from runFunctions import singleImageRun
from storeParameters import storeArray

from utils import createDir
from storeParameters import storeParameters, storePerformace
from poisson import imgToSpikeTrain

networkList = [784, 400]

storeArray("../parameters/assignments.npy", -1*np.ones(networkList[-1]))
storeArray("../parameters/weights1.npy", b2.random(networkList[1]*networkList[0])
+ 0.01)
storeArray("../parameters/theta1.npy", -1*np.ones(networkList[-1]))

from runParameters import *

# Add the parameters of the network to the local variables
from equationsParameters import *
from neuronsParameters import *
locals().update(parametersDict)


# Load the MNIST dataset
imgArray, labelsArray = loadDataset(trainImages, trainLabels)



print(subprocess.run("ls"))

# Create the network data structure
network = createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename, 
		scaleFactors)

for i in range(5):
	# Convert the image into a train of spikes
	imgToSpikeTrain(network, imgArray[i], inputIntensity)


	# Measure the starting time
	startTimeRun = timeit.default_timer()

	# Run the network
	network.run(trainDuration)

	# Measure the ending time
	stopTimeRun = timeit.default_timer()

	print("Total time: ", stopTimeRun - startTimeRun)
