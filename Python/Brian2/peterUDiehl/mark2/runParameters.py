import numpy as np
import sys

equationsDir = "./equations"

if equationsDir not in sys.path:
	sys.path.insert(1, equationsDir)

from equations import *
from parameters import *

networkList = [784, 200, 100]

mode = "train"


trainDuration = 0.35*b2.second
restTime = 0.15*b2.second

equationsDict, stdpDict = defineEquations(mode)

updateInterval = 250
printInterval = 10

startInputIntensity = 2.
inputIntensity = startInputIntensity

accuracies = []
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
