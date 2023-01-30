import subprocess as sp
import numpy as np


def run(networkScript, countersFilename, lastLayerSize, countBitWidth):

	sp.run(networkScript)

	return loadCounters(countersFilename, lastLayerSize, countBitWidth)




def loadCounters(countersFilename, lastLayerSize, countBitWidth):

	with open(countersFilename) as fp:

		spikesCounter = np.zeros(lastLayerSize).astype(int)

		binaryCounters = fp.readline()

		for i in range(lastLayerSize):

			counter = binaryCounters[i*countBitWidth:
					(i+1)*countBitWidth]

			spikesCounter[i] = int(counter, 2)

	return spikesCounter
