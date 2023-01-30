import numpy as np
from parameters import *
from utils import initAssignments

from files import *


# List of layer sizes
networkList = [784, 400]

mode = "test"
trainPrecision = "float"

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
	"exc" 		: 2**(-10)#dt/tauExc,
}


stdpDict["ltp_dt_tau"] = dt/stdpDict["ltp_tau"]
stdpDict["ltd_dt_tau"] = dt/stdpDict["ltd_tau"]


# Array of scale factors for the random generation of the weights
scaleFactors = scaleFactor * np.ones(len(networkList) - 1)


# Update and print intervals expressed in number of images
updateInterval = 10
printInterval = 10


# Initial intensity of the input pixels
startInputIntensity = 2.
inputIntensity = startInputIntensity


# Array of normalization factors for the weights of the various layers 
constSums = constSum * np.ones(len(networkList) - 1)


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
