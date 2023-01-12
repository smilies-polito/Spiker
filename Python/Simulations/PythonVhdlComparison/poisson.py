#!/Users/alessio/anaconda3/bin/python3

import numpy as np


def importSpikes(filename, trainingSteps, inputLayerSize):

	"""
	Read the input spikes from file and convert them into a numpy array.

	INPUT:
		1) filename: string. Name of the file containing the spikes in text
		form.

		2) trainingSteps: total amount of time steps associated to the
		pixels" spikes trains. This must be equal to the number of lines
		in the file.

		3) inputLayerSize: total amount of input spikes per each cycle.
		Each line of the file must contain exactly this amount of "0"
		and "1".

	OUTPUT:
		spikesTrains: two-dimensional numpy array. Time steps on the
		rows, input nodes on the colummns. Input spike sequences.
	"""

	spikesTrains = np.zeros((trainingSteps, inputLayerSize)).astype(bool)
	i = 0

	# Read one single string from the input file
	with open(filename, "r") as fp:

		for line in fp:

			# Remove newline character
			binaryString = line[:-1]

			# Convert string into list
			binaryList = list(binaryString)

			# Convert list of characters into array of integers
			binaryArray = np.array(binaryList).astype(int)

			# Convert integers to boolean
			spikesTrains[i] = binaryArray.astype(bool)
			
			i += 1

	return spikesTrains
	

def imgToSpikeTrain(image, dt, trainingSteps, inputIntensity, rng):

	""" 
	Convert a black and white image into spike trains using the Poisson
	method.

	INPUT:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) dt: time step duration, expressed in milliseconds. 

		3) trainingSteps: total amount of time steps associated to the
		pixels" spikes trains.

		4) inputIntensity: current value of the pixel"s intensity.

		5) rng: NumPy random generator.


	OUTPUT:

		Two-dimensional boolean NumPy array. Each row corresponds to a
		time step. Each column corresponds to a pixel.
	"""

	# Create two-dimensional array of random values
	random2D = rng.uniform(size = (trainingSteps, 1))

	# Convert the image into spikes trains
	return poisson(image, dt, random2D, inputIntensity)





def poisson(image, dt, random2D, inputIntensity):

	""" 
	Poisson convertion of the numerical values of the pixels into spike
	trains. 

	INPUT PARAMETERS:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) dt: time step duration, expressed in milliseconds. 

		3) random2D: teo-dimensional NumPy array containing random
		values between pixelMin and pixelMax.

		4) inputIntensity: current value of the pixel"s intensity.


	OUTPUT:

		Boolean two-dimensional array containing one spikes"
		train for each pixel.  
	"""

	# Convert dt from milliseconds to seconds
	dt = dt*1e-3

	# Create the boolean array of spikes with Poisson distribution
	return ((image*inputIntensity/8.0)*dt)[:] > random2D
