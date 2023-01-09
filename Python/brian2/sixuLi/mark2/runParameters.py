import numpy as np
from equations import defineEquations
from utils import initAssignments
import brian2 as b2

from neuronsParameters import * 


from files import *


# List of layer sizes
networkList = [784, 400]

mode = "train"


# Training and resting periods in milliseconds
trainDuration = 0.35*b2.second
restTime = 0.15*b2.second


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

# Initialize the current and previous counts of the spikes
currentSpikesCount = np.zeros(networkList[-1])
prevSpikesCount = np.zeros(networkList[-1])


# Scaling factor for the random generation of the weights
scaleFactors = np.array([0.3])

# Weights normalization factor
constSum = 78.4

# Initialize the output classification
assignments = initAssignments(mode, networkList, assignmentsFile)

# Select the correct equations depending on the operational mode.
equationsDict, stdpDict = defineEquations(mode)

# Minimum acceptable number of output spikes generated during the training.
countThreshold = 5
