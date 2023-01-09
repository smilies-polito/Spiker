#!/Users/alessio/anaconda3/bin/python3

import numpy as np

# Functions to load data from a generic idx file into a NumPy array of proper
# shape.
#
# IDX FILE FORMAT
#
#	The idx file format consists of:
#	
#		1) A magic number. This is a 32 bit integer value which encodes
#		the type of data contained into the file and the number of
#		dimensions of the stored data structure. 32 bit means 4 byte.
#		The two most significant bytes are always equal to 0 (\x00 in
#		python notation for hexadecimal values).  The two less
#		significant bytes instead encode the information.
#
#			1a) Data type.  Data can be stored in various format
#			into the file.  The type of data is encoded in the
#			second byte (starting the count from the less
#			significant byte of the magic number). The possible
#			values are:
#
#				\x08	->	unsigned byte 
#				\x09	->	byte 
#				\x0B	->	16 bit integer
#				\x0C	->	32 bit integer 
#				\x0D	->	32 bit float
#				\x0E	->	double (64 bit float)
#
#			1b) Number of dimensions of the data structure. Data can
#			be stored with the desired number of dimensions. The
#			minimun dimension is 1 and in this case the data are
#			stored in form of a single array of data. If the
#			dimension is 2 the data are stored in form of an array
#			of arrays, that is a matrix. In the specific case of the
#			MNIST images data are stored as a tridimensional array,
#			that is an array of 28x28 bytes.
#
#		2) An amount of 32 bit integer numbers corresponding to the
#		number of dimensions of the data structure. Each of this integer
#		values represents the number of data stored along a specific
#		dimension.
#
#		3) The data themselves.
#
# The following functions are in charge of decoding the magic number, reading
# the proper amount of integer numbers, interpreting them as the dimensions of
# the data structure, and then reading the data and storing them in form of a
# numpy array.






# Dictionary which is used as hash table to decode the data type from the
# magic number.
# 
# The functions have been developed to work on an Intel-like platform. The
# MNIST dataset stores data in big-endian format (MSB first). On the
# contrary Intel processors read the data in little-endian format. For
# this reason the code corresponding for example to the 32 bit integer
# data type can't be directly decoded with the string "int32" because in
# this way the integer value would be read with the little-endian
# convention, giving a result completely different from the one
# represented in the buffer.  All the numerical values are read using the
# NumPy function frombuffer(). This accepts a string which encodes the
# data type to read from a buffer as one of its arguments. This allows to
# specify the endianness of the data.
# 
# 	">"	: data stored in big-endian format.  
# 	"<"	: data stored in little-endian format.
# 
# Considering the endianness changes the way in which data types are
# expressed. In particular:
# 
# 	"i"	: integer 
# 	"f"	: float
# 
# The data type is followed by a numerical value which expresses the
# number of bytes that compose the data. 	

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
