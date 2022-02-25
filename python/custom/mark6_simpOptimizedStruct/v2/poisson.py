#!/Users/alessio/anaconda3/bin/python3

import numpy as np


def imgToSpikeTrain(image, dt, trainingSteps, inputIntensity, rng):

	''' 
	Convert a black and white image into spike trains using the Poisson
	method.

	INPUT:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) dt: time step duration, expressed in milliseconds. 

		3) trainingSteps: total amount of time steps associated to the
		pixels' spikes trains.

		4) inputIntensity: current value of the pixel's intensity.

		5) rng: NumPy random generator.


	OUTPUT:

		Two-dimensional boolean NumPy array. Each row corresponds to a
		time step. Each column corresponds to a pixel.
	'''

	# Create two-dimensional array of random values
	random2D = rng.uniform(size = (trainingSteps, image.shape[0]))

	# Convert the image into spikes trains
	return poisson(image, dt, random2D, inputIntensity)





def poisson(image, dt, random2D, inputIntensity):

	''' 
	Poisson convertion of the numerical values of the pixels into spike
	trains. 

	INPUT PARAMETERS:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) dt: time step duration, expressed in milliseconds. 

		3) random2D: teo-dimensional NumPy array containing random
		values between pixelMin and pixelMax.

		4) inputIntensity: current value of the pixel's intensity.


	OUTPUT:

		Boolean two-dimensional array containing one spikes'
		train for each pixel.  
	'''

	# Convert dt from milliseconds to seconds
	dt = dt*1e-3

	# Create the boolean array of spikes with Poisson distribution
	return ((image*inputIntensity/8.0)*dt)[:] > random2D
