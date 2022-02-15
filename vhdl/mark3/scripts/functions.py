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




def storeWeights(weightsArray, subArraySize, bitWidth, wordWidth, rootFilename):

	subArraysN = int(weightsArray.shape[0]//subArraySize)
	lastArraySize = int(weightsArray.shape[0]%subArraySize)

	for i in range(subArraysN):

		filename = rootFilename + str(i) + ".mem"

		formatAndStore(weightsArray[i*subArraySize :
			(i+1)*subArraySize].T, bitWidth, wordWidth, filename)

	if lastArraySize > 0:

		filename = rootFilename + str(subArraysN)  + ".mem"	

		formatAndStore(weightsArray[subArraysN*subArraySize :
			subArraysN*subArraySize + lastArraySize].T, bitWidth,
			wordWidth, filename)




def formatAndStore(numpyArray, bitWidth, wordWidth, filename):

	binaryList = arrayToBin(numpyArray, bitWidth, wordWidth)

	storeList(binaryList, filename)


def arrayToBin(numpyArray, bitWidth, wordWidth):

	binaryList = []

	for subArray in numpyArray:

		binaryString = ""

		binaryFormat = "{0:0" + str(bitWidth) + "b}"

		for element in subArray:

			binaryElement = binaryFormat.format(element)

			binaryString = binaryElement + binaryString

		binaryString = "0"*(wordWidth - len(binaryString)) + \
				binaryString

		binaryList.append(binaryString)

	return binaryList



def storeList(inputList, filename):

	with open(filename, 'w') as fp:

		for element in inputList:

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
