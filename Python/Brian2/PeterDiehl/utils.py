import subprocess as sp
import numpy as np

def createDir(dirName):

	'''
	Create a new directory. If it already exists it is firstly remove.

	INPUT:

		dirName: string. Name of the directory to create
	'''

	# Check if the directory exists
	cmdString = "if [[ -d " + dirName + " ]]; then "

	# If it exists remove it
	cmdString += "rm -r " + dirName + "; "
	cmdString += "fi; "

	# Create the directory
	cmdString += "mkdir " + dirName + "; "
	
	# Run the complete bash command
	sp.run(cmdString, shell=True, executable="/bin/bash")


def initAssignments(mode, networkList, assignmentsFile):

	"""
	Associate each output neuron to a label

	INPUT:
		1) mode: string. Can be "train" or "test"

		2) networkList: list. Contains the network structure. Each
		element contain the number of neurons of the corresponding
		layer.

		3) assignmentsFile: string. Name of the file containing the
		pre-trained assignments. Not required if mode = "train"

	OUTOUT:
		Numpy array containing the labels associated to the output
		neurons
	"""

	if mode == "train":
		return -1*np.ones(networkList[-1])

	elif mode == "test":
		with open(assignmentsFile, 'rb') as fp:
			return np.load(fp)

	else:
		print('Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train')
		sys.exit()


def seconds2hhmmss(seconds):

	"""
	Convert seconds into hour minutes seconds.

	INPUT:
		seconds: float. Total number of seconds.

	OUTPUT:
		String. Time in hhmmss format.
	"""

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	seconds = int(seconds % 60)

	return str(hours) + "h " + str(minutes) + "min " + str(seconds) + "s"
