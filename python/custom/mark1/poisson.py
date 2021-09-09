#!/Users/alessio/anaconda3/bin/python3

import numpy as np


def imgToSpikeTrain(image, trainingSteps, pixelMin, pixelMax, inputIntensity):

	''' 
	Convert a black and white image into spike trains using the Poisson
	method.

	INPUT:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) trainingSteps: total amount of time steps associated to the
		pixels' spikes trains.

		3) pixelMin: minimum value of the pixels. In the MNIST the value
		0 is used to express a totally black pixel.

		4) pixelMax: maximum value of the pixels. In the MNIST the value
		255 is used to express a totally white pixel.

		5) inputIntensity: current value of the pixel's intensity.


	OUTPUT:

		Two-dimensional boolean NumPy array. Each row corresponds to a
		time step. Each column corresponds to a pixel.
	'''

	# Create bidimensional array of random values
	rng = np.random.default_rng() 
	random2D = rng.integers(low=pixelMin, high=pixelMax, size = 
			(trainingSteps, image.shape[0]))

	# Convert the image into spikes trains
	return poisson(image, random2D, inputIntensity)





def poisson(image, random2D, inputIntensity):

	''' 
	Poisson convertion of the numerical values of the pixels into spike
	trains. 

	INPUT PARAMETERS:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) random2D: teo-dimensional NumPy array containing random
		values between pixelMin and pixelMax.

		3) inputIntensity: current value of the pixel's intensity.


	OUTPUT:

		Boolean two-dimensional array containing one spikes'
		train for each pixel.  
	'''

	return image*inputIntensity/8 > random2D
