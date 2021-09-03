#!/Users/alessio/anaconda3/bin/python3

import numpy as np


def imgToSpikeTrain(image, timeEvolCycles, N_pixels, pixelMin, pixelMax):

	''' 
	Convert a black and white image into spike trains using the Poisson
	method.

	INPUT PARAMETERS:

		1) image: NumPy array with shape (1, height*width), where height
		and width are the vertical and horizontal dimensions of the
		image expressed in number of pixels. It contains the value of
		each pixel expressed as an integer.

		2) timeEvolCycle: duration of the spike train related to each
		pixel, expressed in time steps.

		3) N_pixels: total number of pixels composing the image,
		corresponding to height*width.

		4) pixelMin: minimum value of the pixels. In the MNIST the value
		0 is used to express a totally black pixel.

		5) pixelMax: maximum value of the pixels. In the MNIST the value
		255 is used to express a totally white pixel.

	OUTPUT VALUES:

		Bidimensional NumPy array containing a train of spikes,
		represented as boolean values (False = no spike, True = spike),
		for each pixel.  
	'''

	# Create bidimensional array of random values
	rng = np.random.default_rng() random2D = rng.integers(low=pixelMin,
		high=pixelMax, size=(timeEvolCycles, N_pixels))

	# Convert the image into spikes trains
	return poisson(image, random2D)





def poisson(image, random2D):

	''' 
	Poisson convertion of the numerical values of the pixels into spike
	trains. 

		INPUT PARAMETERS:

			1) image: NumPy array with shape (1, height*width),
			where height and width are the vertical and horizontal
			dimensions of the image expressed in number of pixels.
			It contains the value of each pixel expressed as an
			integer.

			2) random2D: bidimensional NumPy array with shape
			corresponding to the desired output spike trains
			containing random values between pixelMin and pixelMax.

		RETURN VALUES:

			Boolean two-dimensional array containing one spikes'
			train for each pixel.  
	'''

	return image > random2D
