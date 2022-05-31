import numpy as np
from parameters import *
from utils import initAssignments

from files import *


# List of layer sizes
networkList = [784, 400]

mode = "test"

# List of dictionaries of parameters for the layers
excDictList = [excDict.copy()] * (len(networkList) - 1)


# Arrays of weights for the inter layer connections
inh2excWeights = np.ones(len(networkList) - 1)

# Normalize with respect to the current and previous layer sizes
for i in range(1, len(networkList)):
	inh2excWeights[i-1] = inh2excWeight*refInLayerSize/networkList[i-1]\
			*refCurrLayerSize/networkList[i]

# Training and resting periods in milliseconds
trainDuration = 350	# ms
restTime = 150		# ms


# Time step duration in milliseconds
dt = 0.1		# ms


# Exponential time constants
dt_tauDict = {
	"exc" 		: dt/tauExc,
	"thresh"	: dt/tauThresh
}


stdpParam["ltp_dt_tau"] = dt/stdpParam["ltp_tau"]
stdpParam["ltd_dt_tau"] = dt/stdpParam["ltd_tau"]


# List of dictionaries for the learning parameters
stdpDict = {}

# Normalize with respect to the current and previous layer sizes
for i in range(1, len(networkList)):

    synapseName = "exc2exc" + str(i)

    stdpDict[synapseName] = stdpParam.copy()
    stdpDict[synapseName]["eta_pre"] = stdpDict[synapseName]["eta_pre"]*\
            refInLayerSize/networkList[i-1]*\
            refCurrLayerSize/networkList[i]
    stdpDict[synapseName]["eta_post"] = stdpDict[synapseName]["eta_post"]*\
            refInLayerSize/networkList[i-1]*\
            refCurrLayerSize/networkList[i]


# Array of scale factors for the random generation of the weights
scaleFactors = np.ones(len(networkList) - 1)

# Normalize with respect to the current and previous layer sizes
for i in range(1, len(networkList)):
	scaleFactors[i-1] = scaleFactor*refInLayerSize/networkList[i-1]\
			*refCurrLayerSize/networkList[i]


# Array of normalizing factors
constSums = np.ones(len(networkList) - 1)

# Normalize with respect to the current and previous layer sizes
for i in range(1, len(networkList)):
	constSums[i-1] = constSum*refInLayerSize/networkList[i-1]\
			*refCurrLayerSize/networkList[i]

# Update and print intervals expressed in number of images
updateInterval = 250
printInterval = 10


# Initial intensity of the input pixels
startInputIntensity = 2.
inputIntensity = startInputIntensity

# Initialize history of accuracies
accuracies = []

# Initialize history of spikes
spikesEvolution = np.zeros((updateInterval, networkList[-1]))


# Initialize the output classification
assignments = initAssignments(mode, networkList, assignmentsFile)


# Minimum acceptable number of output spikes generated during the training.
countThreshold = 3

# NumPy default random generator.
rng = np.random.default_rng()
