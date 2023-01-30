#!/Users/alessio/anaconda3/bin/python3

import numpy as np

dictDecoder = {


	# Unsigned byte
	"8"	: "ubyte",
	# Signed byte
	"9"	: "byte",
	# Big-endian 16 bit integer
	"11"	: ">i2",
	# Big-endian 32 bit integer
	"12"	: ">i4",
	# Big-endian 32 bit float
	"13"	: ">f4",
	# Big-endian 64 bit float (double)
	"14"	: ">f8"
}





def loadDataset(images, labels):

	''' 
	Load images and labels from the MNIST dataset into two numpy arrays.
	
		INPUT PARAMETERS:
	
			1) images: string corresponding to the name of the mnist
			file which contains the 28x28 black and white images
			stored in idx format.
	
			2) labels: string corresponding to the name of the mnist
			file which contains the labels, stored as a sequence of
			unsigned bytes.
	
		RETURN VALUES:
	
			1) imgArray: bidimensional NumPy array which contains
			all the images read from the mnist file. Each image is
			stored as a NumPy array with shape (1,784), convenient
			to use it as an input for a neural network in which each
			input neuron corresponds to a pixel.
	
			2) labelsArray: NumPy array which contains all the
			labels read from the mnist file in form of integer
			numbers.
	'''

	# Load the entire content of the file of images into a memory buffer
	imgBuffer = readFile(images)

	# Create the array of images for the training/test
	imgArray = idxBufferToArray(imgBuffer)


	# Load the entire content of the file of labels into a memory buffer
	labelsBuffer = readFile(labels)

	# Create the array of labels for the training/test
	labelsArray = idxBufferToArray(labelsBuffer)

	return imgArray, labelsArray








def readFile(filename):

	'''
	Read the entire content of a binary file and store it in a memory
	buffer.

		INPUT PARAMETERS:
 		
			filename: string corresponding to the name of the file
			to read.

		RETURN VALUES:

			readData: buffer in which the whole content of the file
			is stored as a sequence of bytes.
	'''
	with open(filename, "r+b") as f:
		readData = f.read()

	return readData





	

def idxBufferToArray(buffer):

	'''
	Convert a binary buffer which contains data in idx format to a NumPy
	array.
	
		INPUT PARAMETERS:
	
			buffer: sequence of bytes encoding data in idx format.
			This can be obtained by calling the function readFile()
			to read data from an idx file.
	
		RETURN VALUES:
	
			data: NumPy array containing only the data taken from
			the idx buffer.
	
	The magic number and the dimensions are read and used to determine the
	shape of the output array but they are not present in the output data
	array.
	'''

	# Read and decode the magic number
	dtype, dataDim = magicNumber(buffer)

	# Four byte for the magic number, four byte for each dimension
	offset = 4*dataDim+4

	# Read and store all the dimensions of the data structure
	dimensions = readDimensions(buffer, dataDim)

	# Store the data in a NumPy array of proper shape
	data = loadData(buffer, dtype, offset, dimensions)

	return data





def magicNumber(buffer):

	'''
	Read and decode the magic number.
	
		INPUT PARAMETERS:
	
			buffer: sequence of bytes encoding data in idx format.
			This can be obtained by calling the function readFile()
			to read data from an idx file.
	
		RETURN VALUES:
	
			1) dtype: string expressing the data type. See
			dictDecoder for more details.
	
			2) dataDim: number of dimensions of the data structure
			stored in the idx buffer.
	
	Note that deccodeDataType expects a decimal number as an input, while
	the data type is econded as an hexadecimal value inside the magic
	number. Fortunately the NumPy function frombuffer directly converts the
	hexadecimal values in their decimal counterpart.
	'''

	# Read the magic number as four separated bytes
	mn = np.frombuffer(buffer, dtype="ubyte", count=4)

	# Decode the data type byte
	dtype = decodeDataType(mn[2])

	# Decode the data dimensions byte
	dataDim = mn[3]

	return dtype, dataDim





	

def decodeDataType(intCode):

	'''
	Decode the byte of the magic number which encodes the data type.
	
		INPUT PARAMETERS:
	
			intCode: integer value read from the second less
			significant byte of the magic number, expressed in
			decimal format.
	
		RETURN VALUES:	
	
			The function returns a string corresponding to the data
			type encoded in the magic number. See dictDecoder for
			more details.
	'''
	return dictDecoder[str(intCode)]




	
def readDimensions(buffer, dataDim):

	'''
	Read the proper amount of dimensions from an idx buffer. The dimensions
	are stored as 32 bit integers.
	
		INPUT PARAMETERS:
	
			1) buffer: sequence of bytes encoding data in idx
			format. This can be obtained by calling the function
			readFile() to read data from an idx file.
	
			2) dataDim: number of dimensions to read. This can be
			obtained by calling the function magicNumber().
	
		RETURN VALUES:
	
			The function returns a NumPy with shape (1, dataDim)
			containing all the dimensions.
	'''
	return np.frombuffer(buffer, dtype=">u4", count=dataDim, offset=4)





	

def loadData(buffer, dtype, offset, dimensions):

	'''
	Read the complete data structure from a buffer which stores them in idx
	format.
	
		INPUT PARAMETERS:
	
			1) buffer: sequence of bytes encoding data in idx
			format. This can be obtained by calling the function
			readFile() to read data from an idx file.
	
			2) dtype: string encoding the type of the data to
			read.
	
			3) offset: integer value corresponding to the offset,
			expressed in byte, at which the data start in the idx
			buffer.
	
			4) dimensions: NumPy array containing all the dimensions
			of the data structure to read
	
		RETURN VALUES:
	
			The function returns a NumPy array with the proper shape
			containing all the data.
	'''

	# Store the data in form of a single NumPy array with one dimension
	data = np.frombuffer(buffer, dtype=dtype, count=np.prod(dimensions),
		offset=offset) 

	# Reshape the data in a multidimensional NumPy array and return them
	return reshapeData(data, dimensions)





	

def reshapeData(data, dimensions):

	'''
	Reshape a NumPy array.
	
	The new shape is obtained interpreting the dimensions read from the idx
	buffer. 
	
	If the buffer contained a single dimension it means that the data are
	stored in form of a single array and so also the output NumPy array will
	have a single dimension.
	
	If instead the dimensions are multiple the output array is reshaped in
	form of a bidimensional NumPy array which flattens the higher order
	dimensions in a single one.
	
		INPUT PARAMETER:
	
			1) data: NumPy array containing all the data.
			
			2) dimensions: NumPy array containing all the
			dimensions.
	
		OUTPUT VALUES:
	
			data: reshaped NumPy array.
	'''
	
	# Reshape only if necessary
	if dimensions.size > 1:

		# Reshape into two-dimensional array
		arrayDim = np.prod(dimensions[1:])
		data = np.reshape(data, (dimensions[0], arrayDim))

	return data
