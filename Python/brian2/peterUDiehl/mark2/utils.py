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


def initAssignments(mode, networkList, assignementsFile):

	if mode == "train":
		return -1*np.ones(networkList[-1])

	elif mode == "test":
		with open(assignementsFile, 'rb') as fp:
			return np.load(fp)

	else:
		print('Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train')
		sys.exit()


def seconds2hhmmss(seconds):

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	seconds = int(seconds % 60)

	return str(hours) + "h " + str(minutes) + "min " + str(seconds) + "s"
