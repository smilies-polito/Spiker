import numpy as np
from parameters import *
from utils import initAssignments

from files import *


# List of layer sizes
networkList = [784, 200, 100, 10]

mode = "train"


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



# Initialize the learning rates
stdpDict = {}

for i in range(1, len(networkList)):

	synapseName = "exc2exc" + str(i)

	stdpDict[synapseName] = stdpParam.copy()
	stdpDict[synapseName]["eta_pre"] = stdpDict[synapseName]["eta_pre"]*1
	stdpDict[synapseName]["eta_post"] = stdpDict[synapseName]["eta_post"]*1

# Initialize the excitatory parameters
excDict = {}

for i in range(1, len(networkList)):

	layerName = "excLayer" + str(i)

	excDict[layerName] = excParam.copy()

excDict["excLayer1"]["vThreshPlus"] = 0.05
excDict["excLayer2"]["vThreshPlus"] = 0.05
excDict["excLayer3"]["vThreshPlus"] = 0.01
	

# Array of scale factors for the random generation of the weights
scaleFactors = np.ones(len(networkList) - 1)
scaleFactors[0] = 0.8
scaleFactors[1] = 3
scaleFactors[2] = 7

# Array of normalizing factors
constSums = np.ones(len(networkList) - 1)
constSums[0] = 300
constSums[1] = 500
constSums[2] = 100

# Arrays of weights for the inter layer connections
inh2excWeights = np.ones(len(networkList) - 1)
inh2excWeights[0] = -5
inh2excWeights[1] = -15
inh2excWeights[2] = -10
