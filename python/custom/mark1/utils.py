import subprocess as sp
import numpy as np


def createParamDir(dirName):

	cmdString = "if [[ -d " + dirName + " ]]; then "
	cmdString += "rm -r " + dirName + "; "
	cmdString += "fi; "
	cmdString += "mkdir " + dirName + "; "
	
	sp.Popen(cmdString, shell=True, executable="/bin/bash")




def initAssignments(mode, networkList, assignmentsFile):

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

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	seconds = int(seconds % 60)

	return str(hours) + "h " + str(minutes) + "min " + str(seconds) + "s"



