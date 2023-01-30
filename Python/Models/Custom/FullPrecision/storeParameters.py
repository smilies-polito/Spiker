import numpy as np
import timeit
from utils import seconds2hhmmss


def storeParameters(network, networkList, assignments, weightFilename,
			thresholdFilename, assignmentsFile):

	"""
	Store the parameters of the network inside NumPy files.

	INPUT:

		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) assignments: NumPy array containing one label assignment for
		each output neuron.

		4) weightFilename: string. Root of the weights file name of
		each layer. The function appends the number of the current 
		layer to it.

		5) thresholdFilename: string. Root of the threshold file name
		of each layer. The function appends the number of the current
		layer to it.

		6) assignmentsFile: string. Complete name of the file in which
		the assignments of the output layer will be stored.
	
	"""

	for layer in range(1, len(networkList)):

		# Store the weights of the current synapse
		storeArray(weightFilename + str(layer) + ".npy",
			network["exc2exc" + str(layer)]["weights"])

		# Store the dynamic homeostasis of the current layer
		storeArray(thresholdFilename + str(layer) + ".npy", 
			network["excLayer" + str(layer)]["vThresh"])


	# Store the assignments of the output layer
	storeArray(assignmentsFile, assignments)





def storeArray(filename, numpyArray):

	"""
	Store a NumPy array into a NumPy file.

	INPUT:

		1) filename: string. Name of the file in which the array will be
		stored. The standard requires ".npy" as file extension.

		2) numpyArray: NumPy array that will be stored into the file.
	"""

	with open(filename, "wb") as fp:
		np.save(fp, numpyArray)





def storePerformace(startTimeTraining, accuracies, performanceFile):

	"""
	Store the performance of the network into a text file.

	INPUT:

		1) startTimeTraining: system time corresponfing to the beginning
		of the training.

		2) accuracies: list of strings containing the history of the
		accuracy.

		3) performanceFile: string. Complete name of the file in which
		the performance of the network will be stored.

	"""

	# Format the string with the total elapsed time
	timeString = "Total training time : " + \
		seconds2hhmmss(timeit.default_timer() - startTimeTraining)

	# Format the accuracy values in order to write them one per line
	accuracyString = "Accuracy evolution:\n" + "\n".join(accuracies)

	# Write the strings into the file
	with open(performanceFile, "w") as fp:
		fp.write(timeString)
		fp.write("\n\n")
		fp.write(accuracyString)
