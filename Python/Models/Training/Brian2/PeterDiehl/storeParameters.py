import numpy as np
from utils import seconds2hhmmss
import timeit

def storeParameters(networkList, network, assignments, weightFilename,
			thetaFilename, assignmentsFilename):

	"""
	Store the trained hyper-parameters.

	INPUT:

		1) networkList: list. Contains the network structure. Each
		element contain the number of neurons of the corresponding
		layer.

		2) network: Brian 2 Network object.

		3) assignments: numpy array of integers. Labels associated to
		the output neurons.

		4) weightFilename: string. Name of the file in which to store
		the pre-trained weights.

		5) thetaFilename: string. Name of the file  in which to store
		the pre-trained thresholds.

		6) assignmentsFilename: string. Name of the file in which to
		store the computed output assignments.
		 
	"""

	# Store the input weights in a numpy file
	storeArray(weightFilename + str(1) + ".npy", network["poisson2exc"].w)

	# Store the input thresholds in a numpy file
	storeArray(thetaFilename + str(1) + ".npy", network["excLayer1"].theta)

	for layer in range(2, len(networkList)):

		# Store the weights in a numpy file
		storeArray(weightFilename + str(layer) + ".npy", 
			network["exc2exc" + str(layer)].w)

		# Store the thresholds in a numpy file
		storeArray(thetaFilename + str(layer) + ".npy", 
			network["excLayer" + str(layer)].theta)

	# Store the assignments in a numpy file
	storeArray(assignmentsFilename + ".npy", assignments)




def storeArray(filename, numpyArray):

	"""
	Store a generic numpy array on file.

	INPUT:
		1) filename: string. Name of the file in which to store the
		array.

		2) numpyArray. Numpy array to store.
	"""

	with open(filename, 'wb') as fp:
		np.save(fp, numpyArray)




def storePerformance(startTimeTraining, accuracies, performanceFilename):

	"""
	Store time and accuracy on a text file.

	INPUT:
		1) startTimeTraining: float. Time in which the training/test
		began.

		2) accuracies: list. Temporal evolution of the accuracies.

		3) performanceFilename: string. Name of the target text file.
	"""

	# Measure current time and convert in hours:minutes:seconds
	timeString = "Total training time : " + \
		seconds2hhmmss(timeit.default_timer() - startTimeTraining)

	# Convert list into string to print it on file
	accuracyString = "Accuracy evolution:\n" + "\n".join(accuracies)

	# Print results on text file
	with open(performanceFilename + ".txt", 'w') as fp:
		fp.write(timeString)
		fp.write("\n\n")
		fp.write(accuracyString)
