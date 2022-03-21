import subprocess as sp

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




def storeWeights(weightsArray, subArraySize, bitWidth, wordWidth, rootFilename,
		ext):

	"""
	Store the initializatiom of each BRAM in a dedicated file + store all
	the initializations in a single file (easier to import in VHDL).

	INPUT:

		1) weightsArray: numpy array containing all the weights. One row
		for each neuron. One column for each input.

		2) subArraySize: integer. Number of elements that fit a single BRAM.

		3) bitWidth: intger. Number of bits composing each weight.

		4) wordWidth: integer. Number of bits of the BRAM word.

		5) rootFilename: string. Name of the target file. Will be
		appended with the index of the BRAM and with the file extension.

		6) ext: string. File extension for the generated files.
	"""

	# Number of required BRAMs
	subArraysN = int(weightsArray.shape[0]//subArraySize)

	# Number of elements that fit in the last BRAM
	lastArraySize = int(weightsArray.shape[0]%subArraySize)

	# Single initialization filename
	initFilename = rootFilename + ext

	# Loop over all the BRAMs
	for i in range(subArraysN):

		# Create the name of the single BRAM init file
		filename = rootFilename + str(i) + ext

		# Store the weights of the specific BRAM
		formatAndStore(weightsArray[i*subArraySize :
			(i+1)*subArraySize].T, bitWidth, wordWidth, filename,
			"w")

		# Store the weights of all the BRAM in a single initialization
		# file
		formatAndStore(weightsArray[i*subArraySize :
			(i+1)*subArraySize].T, bitWidth, wordWidth, initFilename,
			"a")

	# Last BRAM
	if lastArraySize > 0:

		# Last BRAMinit filename
		filename = rootFilename + str(subArraysN)  + ".mem"	

		# Store the weights of the last BRAM
		formatAndStore(weightsArray[subArraysN*subArraySize :
			subArraysN*subArraySize + lastArraySize].T, bitWidth,
			wordWidth, filename, "w")

		# Store the weights of all the BRAM in a single initialization
		# file
		formatAndStore(weightsArray[subArraysN*subArraySize :
			subArraysN*subArraySize + lastArraySize].T, bitWidth,
			wordWidth, initFilename, "a")




def formatAndStore(numpyArray, bitWidth, wordWidth, filename, mode):

	"""
	Convert a 2D numpy array into a list of binary strings and store it on a
	file. Each string correspond to one row of the array. The elements of
	the row are translated into binary string with length bitWidth and
	concatenated into a single string with length wordWidth. If wordWidth >
	bitWidth x row size, the remaining bits are set to zero.

	INPUT:

		1) numpyArray: generic numpy array of elements to convert.

		2) bitWidth: integer. Number of bits of the single converted
		element.

		3) wordWidth: integer. Number of bits of the string containing
		the concatenated elements.

		4) filename: string. Name of the output file.

		5) mode: string. Opening mode for the output file.
	"""

	# Convert the numpy array to a list of binary strings
	binaryList = arrayToBin(numpyArray, bitWidth, wordWidth)

	# Store the list of binary strings on the output file
	storeList(binaryList, filename, mode)


def arrayToBin(numpyArray, bitWidth, wordWidth):

	"""
	Convert a 2D numpy array into a list of binary strings. Each string
	corresponds to one row of the array. The elements of the row are
	translated into binary string with length bitWidth and concatenated into
	a single string with length wordWidth. If wordWidth > bitWidth x row
	size, the remaining bits are set to zero.

	INPUT:

		1) numpyArray: generic numpy array of elements to convert.

		2) bitWidth: integer. Number of bits of the single converted
		element.

		3) wordWidth: integer. Number of bits of the string containing
		the concatenated elements.
	"""

	# Initialize an empty list
	binaryList = []

	# Loop over all the rows
	for subArray in numpyArray:

		# Initialize the binary string to empty
		binaryString = ""

		# Set the binary format with the desired bit-width
		binaryFormat = "{0:0" + str(bitWidth) + "b}"

		# Loop over all the element sin the subarray (all the columns)
		for element in subArray:

			# Convert the element into binary format
			binaryElement = binaryFormat.format(element)

			# Append the element to the word
			binaryString = binaryElement + binaryString

		# Set to zero all the remaining bits
		binaryString = "0"*(wordWidth - len(binaryString)) + \
				binaryString

		# Apppend the created binary woerd to the list
		binaryList.append(binaryString)

	return binaryList



def storeList(inputList, filename, mode):

	"""
	Store a list of elements on a file.

	INPUT:

		1) inputList: list of elements to store.

		2) filename: string. Name of the file in which to store the
		list.

		3) mode: string. Opening mode for the file. See python
		documentation for more info.
	"""

	# Open the file in the desired mode
	with open(filename, mode) as fp:

		# Loop over all the elements within the list
		for element in inputList:

			# Write the element on the file
			fp.write(element)
			fp.write("\n")




def fixedPointArray(numpyArray, fixed_point_decimals):

	'''
	Convert a NumPy array into fixed point notation.

	INPUT:

		1) numpyArray: floating point array to convert.

		2) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

	'''

	numpyArray = numpyArray * 2**fixed_point_decimals
	return numpyArray.astype(int)
