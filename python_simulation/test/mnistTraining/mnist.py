#!/Users/alessio/anaconda3/bin/python3

import numpy as np

# Functions to load data from a generic idx file into a NumPy array of proper shape.
#
# IDX FILE FORMAT
#
#	The idx file format consists of:
#	
#		1) A magic number. This is a 32 bit integer value which encodes the type
#		of data contained into the file and the number of dimensions of the
#		stored data structure. 32 bit means 4 byte. The two most significant bytes
#		are always equal to 0 (\x00 in python notation for hexadecimal values).
#		The two less significant bytes instead encode the information.
#
#			1a) Data type. 
#			Data can be stored in various format into the file.
#			The type of data is encoded in the second byte (starting the count
#			from the less significant byte of the magic number). The possible
#			values are:
#
#				\x08	->	unsigned byte
#				\x09	->	byte
#				\x0B	->	16 bit integer
#				\x0C	->	32 bit integer
#				\x0D	->	32 bit float
#				\x0E	->	double (64 bit float)
#
#			1b) Number of dimensions of the data structure. Data can be stored
#			with the desired number of dimensions. The minimun dimension is 1
#			and in this case the data are stored in form of a single array of
#			data. If the dimension is 2 the data are stored in form of an
#			array of arrays, that is a matrix. In the specific case of the
#			MNIST images data are stored as a tridimensional array, that is 
#			an array of 28x28 bytes.
#
#		2) An amount of 32 bit integer numbers corresponding to the number of
#		dimensions of the data structure. Each of this integer values represents
#		the number of data stored along a specific dimension.
#
#		3) The data themselves.
#
# The following functions are in charge of decoding the magic number, reading the proper
# amount of integer numbers, interpreting them as the dimensions of the data structure,
# and then reading the data and storing them in form of a numpy array with the same
# dimensions.





# Dictionary which is used as hash table to decode the data type from the magic number
dictDecoder = {

	"8"	: "ubyte",
	"9"	: "byte",
	"11"	: ">i2",
	"12"	: ">i4",
	"13"	: ">f4",
	"14"	: ">f8"

}



# Load images and labels from the MNIST dataset into two numpy arrays.
#
# 	INPUT PARAMETERS:
#
# 		1) images: string corresponding to the name of the mnist file which
# 		contains the 28x28 black and white images stored in idx format.
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
