from mnist import loadDataset
from createNetwork import createNetwork

from files import *
from trainingParameters import *


# Load the MNIST dataset
imgArray, labelsArray = loadDataset(images, labels)

# Create the network data structure
network = createNetwork(networkList, weightFilename, thetaFilename, mode, 
			excDictList, inhDictList, scaleFactors, exc2inhWeights,
			inh2excWeights)
















# i = 0
# 
# startTimeTraining = timeit.default_timer()
# 
# numberOfCycles = imgArray.shape[0]
# 
# while i < numberOfCycles:
# 
# 	inputIntensity, i, accuracies = trainTestCycle(imgArray[i], networkList, 
# 		network, singleExampleTime, restTime, spikesEvolution, 
# 		updateInterval, printInterval, currentSpikesCount, 
# 		prevSpikesCount, startTimeTraining, accuracies, labelsArray, 
# 		assignements,inputIntensity, startInputIntensity, i, mode,
# 		constSum)
# 
# 
# storeParameters(networkList, network, assignements, weightFilename, 
# 		thetaFilename, assignementsFilename)
# 
# storePerformace(startTimeTraining, accuracies, performanceFilename)
