import numpy as np
import torch
from parameters import *
from utils import initAssignments

from files import *


# dataloader arguments
batch_size = 256

dtype = torch.float

# Network Architecture
num_inputs = 28*28
num_hidden = 128
num_outputs = 10

# Temporal Dynamics
num_steps = 100
beta = 0.9375

num_epochs = 30
loss_hist = []
test_loss_hist = []
counter = 0

# List of layer sizes
networkList = [784, 128, 10]

mode = "test"
trainPrecision = "float"

# List of dictionaries of parameters for the layers
excDictList = [excDict] * (len(networkList) - 1)

# Training and resting periods in milliseconds
trainDuration = 350	# ms

# Array of scale factors for the random generation of the weights
scaleFactors = np.ones(len(networkList) - 1)



# Time step duration in milliseconds
dt = 2**-4 * tauExc		# ms


# Exponential time constants
dt_tauDict = {
	"exc" 		: dt/tauExc,
}

# Update and print intervals expressed in number of images
updateInterval = 10
printInterval = 250


# Initial intensity of the input pixels
startInputIntensity = 2.
inputIntensity = startInputIntensity


# Initialize history of accuracies
accuracies = []

# Initialize history of spikes
spikesEvolution = np.zeros((updateInterval, networkList[-1]))


# Minimum acceptable number of output spikes generated during the training.
countThreshold = 5

# NumPy default random generator.
rng = np.random.default_rng()
