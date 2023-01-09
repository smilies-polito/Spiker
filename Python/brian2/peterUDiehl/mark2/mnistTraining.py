import numpy as np
import brian2 as b2
import timeit

from mnist import loadDataset
from createNetwork import createNetwork
from trainTestFunctions import *
from utils import createDir, initAssignments
from files import *
from runParameters import *


assignements = initAssignments(mode, networkList, assignementsFile)

imgArray, labelsArray = loadDataset(images, labels)



network = createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename, 
		scaleFactors)


i = 0

startTimeTraining = timeit.default_timer()

numberOfCycles = imgArray.shape[0]

while i < numberOfCycles:

	inputIntensity, i, accuracies = trainTestCycle(imgArray[i], networkList, 
		network, trainDuration, restTime, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeTraining, accuracies, labelsArray, 
		assignements,inputIntensity, startInputIntensity, i, mode)


createDir(paramDir)

storeParameters(networkList, network, assignements, weightFilename, 
		thetaFilename, assignementsFilename)

storePerformace(startTimeTraining, accuracies, performanceFilename)
