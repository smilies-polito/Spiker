import numpy as np
import sys

equationsDir = "./equations"

if equationsDir not in sys.path:
	sys.path.insert(1, equationsDir)

from equations import *
from parameters import *

# Network structure. One element for each layer = number of nodes of that layer
networkList = [784, 400]

# Execution mode. Can be "train" or "test"
mode = "train"

# Temporal window associated to an input image
trainDuration = 0.35*b2.second

# Temporal window associated to the rest after one pass over an image
restTime = 0.15*b2.second

# Select proper equations depending on the execution mode
equationsDict, stdpDict = defineEquations(mode)

# Number of images after which the network parameters are updated and the
# accuracy is evaluated
updateInterval = 100

# Number of images after which the accuracy evolution is printed
printInterval = 10

# Initial input intensity
startInputIntensity = 2.

# Variable input intensity
inputIntensity = startInputIntensity

# List of accuracies
accuracies = []

# Preallocate the monitors
spikesEvolution = np.zeros((updateInterval, networkList[-1]))
currentSpikesCount = np.zeros(networkList[-1])
prevSpikesCount = np.zeros(networkList[-1])

# Array of scale factors for the random generation of the weights
scaleFactors = np.ones(len(networkList) - 1)
for i in range(1, len(networkList)):
	scaleFactors[i-1] = scaleFactor*refInLayerSize/networkList[i-1]\
			*refCurrLayerSize/networkList[i]

# Array of normalizing factors
constSums = np.ones(len(networkList) - 1)
for i in range(1, len(networkList)):
	constSums[i-1] = constSum*refInLayerSize/networkList[i-1]\
			*refCurrLayerSize/networkList[i]
