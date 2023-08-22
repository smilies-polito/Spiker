import subprocess as sp
import numpy as np
from random import randint


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


def ceil_pow2(x):

	power = 0
	go = True

	while go:
		if x // 2**power == 0 or x // 2**power == 1 and \
			x % 2**power == 0:

			go = False

		else:
			power += 1
			
	return 2**power

def n_bytes(x):

	if x == 0:
		return 1

	return int((x-1) // 8) + 1


def floor_pow2(x):

	power = 0
	go = True

	while go:
		if x // 2**power == 1 or x == 0:
			go = False

		else:
			power += 1
			
	return 2**power


def random_binary(min_value = 0, max_value = 255, bitwidth = 8):

	if max_value > 2**bitwidth - 1 or min_value < 0:
		print("Random number not representable on bitwidth\n")
		exit(-1)

	rand_int = randint(min_value, max_value)

	return "{0:{fill}{width}{base}}".format(rand_int, fill = 0,
		width = bitwidth, base = "b")

def int_to_bin(value, fill = 0, width = 8):
	return "{0:{fill}{width}{base}}".format(value, fill = fill, width =
			width, base = "b")

def int_to_hex(value, fill = 0, width = 8):
	return "{0:{fill}{width}{base}}".format(value, fill = fill, width =
			width, base = "x")

def fixed_point_array(numpyArray, bitwidth, fixed_point_decimals = 0,
	conv_type = "signed"):

	'''
	Convert a NumPy array into fixed point notation.

	INPUT:

		1) numpyArray: floating point array to convert.

		2) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

	'''

	numpyArray = numpyArray * 2**fixed_point_decimals
	fp_array = numpyArray.astype(int)

	if conv_type == "signed":
		fp_array[fp_array > 2**(bitwidth-1)-1] = 2**(bitwidth-1)-1
		fp_array[fp_array < -2**(bitwidth-1)] = -2**(bitwidth-1)

	elif conv_type == "unsigned":
		fp_array[fp_array > 2**bitwidth-1] = 2**(bitwidth)-1

	else:
		raise ValueError("Invalid fixed-point convertion type")

	return fp_array
