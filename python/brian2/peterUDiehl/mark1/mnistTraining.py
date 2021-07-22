import numpy as np
import brian2 as b2
import timeit

from equations import *
from equationsParameters import *
from neuronsParameters import *
from mnist import loadDataset
from createNetwork import createNetwork
from trainFunctions import *
from utils import createParamDir


images = "../../mnist/t10k-images-idx3-ubyte"
labels = "../../mnist/t10k-labels-idx1-ubyte"

paramDir = "./parameters"

weightFilename = paramDir + "/weights"
thetaFilename = paramDir + "/theta"
performanceFilename = paramDir + "/performance"
assignementsFile = paramDir + "/assignements"


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


i = 0

startTimeTraining = timeit.default_timer()

numberOfCycles = 502 #imgArray.shape[0]

while i < numberOfCycles:

	inputIntensity, i, accuracies = trainCycle(imgArray[i], networkList, 
		network, singleExampleTime, restTime, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeTraining, accuracies, labelsArray, 
		assignements,inputIntensity, startInputIntensity, i, mode)


storeParameters(networkList, network, assigmenets, weightFilename, 
		thetaFilename, assignementsFilename)

storePerformace(startTimeTraining, accuracies, performanceFilename)
