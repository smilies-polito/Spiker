import numpy as np
from evaluationParameters import *

from files import *


# List of layer sizes
networkList = [784, 400]

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


stdpParam["ltp_dt_tau"] = dt/stdpParam["ltp_tau"]
stdpParam["ltd_dt_tau"] = dt/stdpParam["ltd_tau"]


# List of dictionaries for the learning parameters
stdpDict = {}

# Normalize with respect to the current and previous layer sizes
for i in range(1, len(networkList)):

    synapseName = "exc2exc" + str(i)

    stdpDict[synapseName] = stdpParam.copy()



# Array of scale factors for the random generation of the weights
scaleFactors = scaleFactor * np.ones(len(networkList) - 1)


# Update and print intervals expressed in number of images
updateInterval = 250
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


# Minimum acceptable number of output spikes generated during the training.
countThreshold = 5

# NumPy default random generator.
rng = np.random.default_rng()
