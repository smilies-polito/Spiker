#!/Users/alessio/anaconda3/bin/python3

import sys

development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/parallelTraining"

if development not in sys.path:
	sys.path.insert(1,development)


from trainSnn import trainSnn
from poisson import imgToSpikeTrain

import numpy as np


def train(images, labels, N_subsets, timeEvolCycles, N_pixels, pixelMin, pixelMax,
		labelsArray, networkDictList, v_mem_dt_tau, stdp_dt_tau, v_reset, A_ltp, 
		A_ltd, classificationArray):

		subsetLength = int(images.size // N_subsets)
		lastSubsetLength = images.size % N_subsets

		for i in range(N_subsets-1):
			
			accuracy = trainSubset(images, labels, subsetLength, 
					timeEvolCycles, N_pixels, pixelMin, pixelMax, 
					labelsArray, networkDictList, v_mem_dt_tau, 
					stdp_dt_tau, v_reset, A_ltp, A_ltd, 
					classificationArray)

			logString = "Accuracy: " + str(accuracy) + ", Labels' array: " + \
					str(labelsArray)
			print(logString)

		accuracy = trainSubset(images, labels, lastSubsetLength, 
					timeEvolCycles, N_pixels, pixelMin, pixelMax, 
					labelsArray, networkDictList, v_mem_dt_tau, 
					stdp_dt_tau, v_reset, A_ltp, A_ltd, 
					classificationArray)

		logString = "Accuracy: " + str(accuracy) + ", Labels' array: " + \
					str(labelsArray)
		print(logString)





def trainSubset(images, labels, subsetLength, timeEvolCycles, N_pixels, pixelMin, pixelMax,
		labelsArray, networkDictList, v_mem_dt_tau, stdp_dt_tau, v_reset, A_ltp, 
		A_ltd, classificationArray):

		accuracy = 0

		for i in range(subsetLength):

			poissonImg = imgToSpikeTrain(images[i], timeEvolCycles, N_pixels,
					pixelMin, pixelMax)

			label = labels[i]

			classResult = trainSingleImg(poissonImg, label, labelsArray,
					networkDictList, v_mem_dt_tau, stdp_dt_tau,
					v_reset, A_ltp, A_ltd, 
					classificationArray[label])

			accuracy = accuracy + classResult

			restartNetwork(networkDictList)	

			

		return accuracy/subsetLength





def restartNetwork(networkDictList):

	for i in range(len(networkDictList)):

		restartLayer(networkDictList[i])





def restartLayer(layerDict):
	
	layerDict["v_mem"][:] = 0
	layerDict["t_in"][:] = 0
	layerDict["t_out"][:] = 0
	layerDict["outEvents"][:] = 0





def trainSingleImg(poissonImg, label, labelsArray, networkDictList, v_mem_dt_tau,
			stdp_dt_tau, v_reset, A_ltp, A_ltd, spikeCountArray):


	for i in range(len(poissonImg)):

		trainSnn(poissonImg[i], networkDictList, v_mem_dt_tau, stdp_dt_tau,
				v_reset, A_ltp, A_ltd, i)
		updateSpikeCount(spikeCountArray, networkDictList[-1]["outEvents"])

	return accuracyAndClassification(spikeCountArray, labelsArray, label)



def updateSpikeCount(spikeCountArray, outEvents):
	spikeCountArray[outEvents] +=1



def accuracyAndClassification(spikeCountArray, labelsArray, label):

	if label in labelsArray[spikeCountArray == np.max(spikeCountArray)]:
		return 1

	else:
		updateClassification(spikeCountArray, labelsArray, label)
		return 0



def updateClassification(spikeCountArray, labelsArray, label):
	
	mfn = mostFiringNeuron(spikeCountArray)

	ln = labelNeuron(labelsArray, label)

	labelsArray[mfn], labelsArray[ln] = labelsArray[ln], labelsArray[mfn]


def mostFiringNeuron(spikeCountArray):
	return np.argmax(spikeCountArray)


def labelNeuron(labelsArray, label):
	return np.where(labelsArray==label)[0][0]

