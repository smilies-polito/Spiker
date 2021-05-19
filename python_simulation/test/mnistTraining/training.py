#!/Users/alessio/anaconda3/bin/python3

import sys

development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/parallelTraining"

if development not in sys.path:
	sys.path.insert(1,development)


from trainSnn import trainSnn

import numpy as np


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

