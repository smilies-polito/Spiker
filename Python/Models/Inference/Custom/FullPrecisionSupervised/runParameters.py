import torch
import numpy as np

from poisson import imgToSpikeTrain

from parameters import *
from files import *

# dataloader arguments
batch_size = 256

dtype = torch.float

# Network Architecture
num_inputs = 28*28
num_hidden = 400
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.9375

num_epochs = 100
loss_hist = []
test_loss_hist = []
counter = 0

# List of layer sizes
networkList = [784, 400, 10]

# List of dictionaries of parameters for the layers
excDictList = [excDict.copy()] * (len(networkList) - 1)

# Arrays of weights for the inter layer connections
inh2excWeights = np.ones(len(networkList) - 1)


# Time step duration in milliseconds
dt = 2.0**(-4) * 100		# ms

num_steps = 25

# Exponential time constants
dt_tauDict = {
	"exc" 		: dt/tauExc,
}

# Array of scale factors for the random generation of the weights
scaleFactors = np.ones(len(networkList) - 1)


# Update and print intervals expressed in number of images
printInterval = 10
updateInterval = 250

# Initialize history of accuracies
accuracies = []

# Initialize history of spikes
spikesEvolution = np.zeros((updateInterval, networkList[-1]))

# Minimum acceptable number of output spikes generated during the training.
countThreshold = 3

# NumPy default random generator.
rng = np.random.default_rng()


for i in range(1, len(networkList)):
	thresholds = np.ones((1, networkList[i]))*excDict["vThresh0"]
	np.save(thresholdsFilename + str(i) + ".npy", thresholds)
