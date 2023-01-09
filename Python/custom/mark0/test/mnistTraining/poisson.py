#!/Users/alessio/anaconda3/bin/python3

import numpy as np


# Convert a black and white image into spike trains using the Poisson method.
#
# 	INPUT PARAMETERS:
#
# 		1) image: NumPy array with shape (1, height*width), where height and width
# 		are the vertical and horizontal dimensions of the image expressed in
# 		number of pixels. It contains the value of each pixel expressed as an
# 		integer.
#
# 		2) timeEvolCycle: duration of the spike train related to each pixel,
# 		expressed in time steps.
#
# 		3) N_pixels: total number of pixels composing the image, corresponding to
# 		height*width.
#
# 		4) pixelMin: minimum value of the pixels. In the MNIST the value 0 is used
# 		to express a totally black pixel.
#
# 		5) pixelMax: maximum value of the pixels. In the MNIST the value 255 is
# 		used to express a totally white pixel.
#
# 	OUTPUT VALUES:
#
# 		The function returns a bidimensional NumPy array containing a train of
# 		spikes, represented as boolean values (False = no spike, True = spike),
# 		for each neuron. Each element of the array corresponds to a time step and
#		contains a value for each neuron.

def imgToSpikeTrain(image, timeEvolCycles, N_pixels, pixelMin, pixelMax):

	rng = np.random.default_rng()

	random2D = rng.integers(low=pixelMin, high=pixelMax, size=(timeEvolCycles, 
			N_pixels))

	return poisson(image, random2D)






# Poisson convertion of the numerical values of the pixels into spike trains. 
#
# The function exploits the Poisson method which consists in generating a randomly
# distributed train of spikes using the numerical value of each pixel as a firing rate
# which affects the probability of emitting a spike. In this way the higher the pixel
# value, the higher the firing rate and so a the larger the amount of spikes within the
# selected time duration of the train 
#
# 	INPUT PARAMETERS:
#
# 		1) image: NumPy array with shape (1, height*width), where height and width
# 		are the vertical and horizontal dimensions of the image expressed in
# 		number of pixels. It contains the value of each pixel expressed as an
# 		integer.
#
# 		2) random2D: bidimensional NumPy array with shape corresponding to the
# 		desired output spike trains containing random values between pixelMin and
# 		pixelMax.
#
# 	RETURN VALUES:
#
# 		The function simply compares the image with the random numbers for each
# 		time step. If the pixel value is higher then the random one then it emits 
# 		a spike, which corresponds to a True value in the output array. Otherwise
# 		the value is set to False.

def poisson(image, random2D):
	return image > random2D
