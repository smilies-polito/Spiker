def storeWeights(weightsArray, subArraySize, bitWidth, wordWidth, rootFilename):

	subArraysN = int(weightsArray.shape[0]//subArraySize)
	lastArraySize = int(weightsArray.shape[0]%subArraySize)

	for i in range(subArraysN):

		filename = rootFilename + str(i) + ".mem"

		formatAndStore(weightsArray[i*subArraySize :
			(i+1)*subArraySize], bitWidth, wordWidth, filename)

	if lastArraySize > 0:

		filename = rootFilename + str(subArraysN)  + ".mem"	

		formatAndStore(weightsArray[subArraysN*subArraySize :
			subArraysN*subArraySize + lastArraySize], bitWidth,
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
