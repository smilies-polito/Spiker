
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

			binaryString += binaryElement

		binaryString = "0"*(wordWidth - len(binaryString)) + \
				binaryString

		binaryList.append(binaryString)

	return binaryList



def storeList(inputList, filename):

	with open(filename, 'w') as fp:

		for element in inputList:

			fp.write(element)
			fp.write("\n")


import numpy as np

numpyArray = np.array([[1, 4], [2, 5], [3, 6]])
bitWidth = 4
wordWidth = 10
filename = "weights.mem"

formatAndStore(numpyArray, bitWidth, wordWidth, filename)
