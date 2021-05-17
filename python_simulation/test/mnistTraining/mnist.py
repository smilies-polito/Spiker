#!/Users/alessio/anaconda3/bin/python3

import numpy as np

dictDecoder = {

	"8"	: "ubyte",
	"9"	: "byte",
	"11"	: ">i2",
	"12"	: ">i4",
	"13"	: ">f4",
	"14"	: ">f8"

}


def loadDataset(images, labels):
	
	imgBuffer = readFile(images)
	imgArray = idxBufferToArray(imgBuffer)

	labelsBuffer = readFile(labels)
	labelsArray = idxBufferToArray(labelsBuffer)

	return imgArray, labelsArray



	
def readFile(filename):

	with open(filename, "r+b") as f:
		readData = f.read()

	return readData



def idxBufferToArray(buffer):

	dtype, dataDim = magicNumber(buffer)

	offset = 4*dataDim+4

	dimensions = readDimensions(buffer, dataDim)

	data = loadData(buffer, dtype, offset, dimensions)

	return data



def magicNumber(buffer):

	mn = np.frombuffer(buffer, dtype="ubyte", count=4)

	dtype = decodeDataType(mn[2])
	dataDim = mn[3]

	return dtype, dataDim


def decodeDataType(intCode):
	return dictDecoder[str(intCode)]



def readDimensions(buffer, dataDim):
	return np.frombuffer(buffer, dtype=">u4", count=dataDim, offset=4)





def loadData(buffer, dtype, offset, dimensions):

	data = np.frombuffer(buffer, dtype=dtype, count=np.prod(dimensions),
		offset=offset) 

	return reshapeData(data, dimensions)




def reshapeData(data, dimensions):
	
	if dimensions.size > 1:
		arrayDim = np.prod(dimensions[1:])
		data = np.reshape(data, (dimensions[0], arrayDim))

	return data
