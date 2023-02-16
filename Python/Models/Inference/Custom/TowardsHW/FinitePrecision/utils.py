import subprocess as sp
import numpy as np
import sys


def createDir(dirName):

	"""
	Create a new directory. If it already exists it is firstly remove.

	INPUT:

		dirName: string. Name of the directory to create
	"""

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
	Initialize the assignments of the output layer"s neurons.

	INPUT:
		
		1) mode: string. It can be "train" or "test".

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) assignmentsFile: string. Complete name of the file which
		contains the assignments of the output layer.

	"""

	if mode == "train":

		# Initialize assignments to value different from all the labels
		return -1*np.ones(networkList[-1])

	elif mode == "test":

		# Load the assignments from file	
		with open(assignmentsFile, "rb") as fp:
			return np.load(fp)

	else:
		print("Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train")
		sys.exit()


def seconds2hhmmss(seconds):

	"""
	Convert a time value from seconds to hh.mm.ss format

	INPUT:
		
		seconds: float number. Total amount of seconds to convert.

	OUTPUT:

		string containing the time expressed in hh.mm.ss format
	"""

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	seconds = int(seconds % 60)

	return str(hours) + "h " + str(minutes) + "min " + str(seconds) + "s"




def expDecay(dictionary, key, exp_shift, variable): 
	
	""" 
	Decrease the desired integer variable belonging to an entry of the dictionary
	with exponential decay.

	INPUT: 

		1) dictionary: generic dictionary of dictionaries.

		2) key: string. Name of the dictionary entry toupdate		

		3) exp_shift: bit shift for the exponential decay.

		4) variable: string. Name of the variable to update. This is the
		key of dictionary[key].
	"""

	dictionary[key][variable] -= dictionary[key][variable] >> exp_shift




def fixedPoint(value, fixed_point_decimals):

	"""
	Convert a value into fixed point notation.

	INPUT:

		1) value: floating point value to convert.

		2) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

	"""
	return int(value * 2**fixed_point_decimals)




def fixedPointArray(numpyArray, fixed_point_decimals):

	"""
	Convert a NumPy array into fixed point notation.

	INPUT:

		1) numpyArray: floating point array to convert.

		2) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

	"""

	numpyArray = numpyArray * 2**fixed_point_decimals
	return numpyArray.astype(int)



def checkBitWidth(numpyArray, bitWidth):

	"""
	Check that values inside NumPy array don"t exceed a threshold.

	INPUT:

		1) numpyArray: array of values to check.

		2) bitWidth: number of bits on which the neuron works.
	"""
	

	if (numpyArray > 2**(bitWidth-1)-1).any():
		print("Value too high")
		sys.exit()

	elif (numpyArray < -2**(bitWidth-1)).any():
		print("Value too low")
		sys.exit()
