import numpy as np
import brian2 as b2
import timeit

from equations import *
from equationsParameters import *
from neuronsParameters import *
from mnist import loadDataset
from createNetwork import createNetwork
from trainTestFunctions import *
from utils import createParamDir


images = "../../mnist/train-images-idx3-ubyte"
labels = "../../mnist/train-labels-idx1-ubyte"

paramDir = "./parameters"

weightFilename = paramDir + "/weights"
thetaFilename = paramDir + "/theta"
performanceFilename = paramDir + "/performance"
assignementsFilename = paramDir + "/assignements"
assignementsFile = paramDir + "/assignements.npy"




createParamDir(paramDir)


networkList = [784, 400]

mode = "train"


updateInterval = 250
printInterval = 10
startInputIntensity = 2.
inputIntensity = startInputIntensity
scaleFactors = np.array([0.3])


accuracies = []
spikesEvolution = np.zeros((updateInterval, networkList[-1]))
currentSpikesCount = np.zeros(networkList[-1])
prevSpikesCount = np.zeros(networkList[-1])


singleExampleTime = 0.35*b2.second
restTime = 0.15*b2.second

assignements = initAssignements(mode, networkList, assignementsFile)

imgArray, labelsArray = loadDataset(images, labels)

equationsDict, stdpDict = defineEquations(mode)


network = createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename, 
		scaleFactors)


imgArray = np.repeat(imgArray, 2, axis=0)
labelsArray = np.repeat(labelsArray, 2, axis=0)

i = 0

startTimeTraining = timeit.default_timer()

numberOfCycles = imgArray.shape[0]

while i < numberOfCycles:

	inputIntensity, i, accuracies = trainTestCycle(imgArray[i], networkList, 
		network, singleExampleTime, restTime, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeTraining, accuracies, labelsArray, 
		assignements,inputIntensity, startInputIntensity, i, mode)


storeParameters(networkList, network, assignements, weightFilename, 
		thetaFilename, assignementsFilename)

storePerformace(startTimeTraining, accuracies, performanceFilename)
