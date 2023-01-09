import numpy as np
from utils import seconds2hhmmss
import timeit

def storeParameters(networkList, network, assignements, weightFilename,
			thetaFilename, assignementsFilename):

	storeArray(weightFilename + str(1) + ".npy", network["poisson2exc"].w)

	storeArray(thetaFilename + str(1) + ".npy", network["excLayer1"].theta)

	for layer in range(2, len(networkList)):

		storeArray(weightFilename + str(layer) + ".npy", 
			network["exc2exc" + str(layer)].w)

		storeArray(thetaFilename + str(layer) + ".npy", 
			network["excLayer" + str(layer)].theta)


	storeArray(assignementsFilename + ".npy", assignements)




def storeArray(filename, numpyArray):

	with open(filename, 'wb') as fp:
		np.save(fp, numpyArray)




def storePerformance(startTimeTraining, accuracies, performanceFilename):

	timeString = "Total training time : " + \
		seconds2hhmmss(timeit.default_timer() - startTimeTraining)

	accuracyString = "Accuracy evolution:\n" + "\n".join(accuracies)

	with open(performanceFilename + ".txt", 'w') as fp:
		fp.write(timeString)
		fp.write("\n\n")
		fp.write(accuracyString)
