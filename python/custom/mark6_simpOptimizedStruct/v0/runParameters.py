import numpy as np
from parameters import *
from utils import initAssignments

from files import *


# List of layer sizes
networkList = [784, 200, 100, 10]

mode = "train"

# List of dictionaries of parameters for the layers
excDictList = [excDict] * (len(networkList) - 1)


# Arrays of weights for the inter layer connections
inh2excWeights = inh2excWeight * np.ones(len(networkList) - 1)


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


stdpDict["ltp_dt_tau"] = dt/stdpDict["ltp_tau"]
stdpDict["ltd_dt_tau"] = dt/stdpDict["ltd_tau"]


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

# Update and print intervals expressed in number of images
updateInterval = 10
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
countThreshold = 5

# NumPy default random generator.
rng = np.random.default_rng()

# Lables. Used to generalize and use arbitrary labels
labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
