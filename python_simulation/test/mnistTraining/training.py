#!/Users/alessio/anaconda3/bin/python3

import sys

development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/parallelTraining"

if development not in sys.path:
	sys.path.insert(1,development)


from trainSnn import trainSnn
from simFunctions import addToTemporalEvolution

import numpy as np


def trainSingleImg(poissonImg, label, labelsArray, networkDictList, dt_tau, v_reset, 
			A_ltp, A_ltd, spikeCountArray,v_memEvol_list, outEventsEvol_list, weightsEvol_list):


	for i in range(len(poissonImg)):

		trainSnn(poissonImg[i], networkDictList, dt_tau, v_reset, A_ltp, A_ltd, i)
		updateSpikeCount(spikeCountArray, networkDictList[-1]["outEvents"])

		# To remove once tested
		addToTemporalEvolution(networkDictList, "v_mem", v_memEvol_list, i)

		addToTemporalEvolution(networkDictList, "outEvents", 
					outEventsEvol_list, i)

		addToTemporalEvolution(networkDictList, "weights", weightsEvol_list, i)

	print("\nSpike Count Array")
	print(spikeCountArray)
	print("\n")

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

