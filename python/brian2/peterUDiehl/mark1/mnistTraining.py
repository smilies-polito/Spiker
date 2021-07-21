import numpy as np
import brian2 as b2
import timeit

from equations import *
from equationsParameters import *
from neuronsParameters import *
from mnist import loadDataset
from createNetwork import createNetwork
from trainFunctions import trainCycle


images = "../../mnist/train-images-idx3-ubyte"
labels = "../../mnist/train-labels-idx1-ubyte"


networkList = [784, 400]

mode = "test"
weightFilename = "weights"
thetaFilename = "theta"

weightFile = "weights1.txt"
thetaFile = "theta1.txt"





updateInterval = 250
printInterval = 10
startInputIntensity = 2.
inputIntensity = startInputIntensity


accuracies = []
spikesEvolution = np.zeros((updateInterval, networkList[-1]))
currentSpikesCount = np.zeros(networkList[-1])
prevSpikesCount = np.zeros(networkList[-1])
assignements = -1*np.ones(networkList[-1])


singleExampleTime = 0.35*b2.second
restTime = 0.15*b2.second


imgArray, labelsArray = loadDataset(images, labels)

equationsDict, stdpDict = defineEquations(mode)


with open(thetaFile, 'wb') as fp:
	np.save(fp, np.random.randn(networkList[-1]))

with open(weightFile, 'wb') as fp:
	np.save(fp, np.random.randn(networkList[-1]*networkList[0]))

network = createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename)


i = 0

startTimeTraining = timeit.default_timer()

while i < imgArray.shape[0]:

	inputIntensity, i, accuracies = trainCycle(imgArray[i], networkList, 
		network, singleExampleTime, restTime, spikesEvolution, 
		updateInterval, printInterval, currentSpikesCount, 
		prevSpikesCount, startTimeTraining, accuracies, labelsArray, 
		assignements,inputIntensity, startInputIntensity, i)
