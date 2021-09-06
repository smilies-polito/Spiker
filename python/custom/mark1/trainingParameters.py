import numpy as np
from networkParameters import *

networkList = [784, 400]

mode = "train"

excDictList = [excDict] * (len(networkList) - 1)
inhDictList = [inhDict] * (len(networkList) - 1)

exc2inhWeights = exc2inhWeight * np.ones(len(networkList) - 1)
inh2excWeights = inh2excWeight * np.ones(len(networkList) - 1)

scaleFactors = np.array([10])


updateInterval = 250
printInterval = 10
startInputIntensity = 2.
inputIntensity = startInputIntensity

#singleExampleTime = 0.35*b2.second
#restTime = 0.15*b2.second
constSum = 78.4


accuracies = []
spikesEvolution = np.zeros((updateInterval, networkList[-1]))
currentSpikesCount = np.zeros(networkList[-1])
prevSpikesCount = np.zeros(networkList[-1])


# assignements = initAssignements(mode, networkList, assignementsFile)
