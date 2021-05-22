#!/Users/alessio/anaconda3/bin/python3

import numpy as np


def imgToSpikeTrain(image, timeEvolCycles, N_pixels, pixelMin, pixelMax):

	rng = np.random.default_rng()

	random2D = rng.integers(low=pixelMin, high=pixelMax, size=(timeEvolCycles, 
			N_pixels))

	return poisson(image, random2D)


def poisson(image, random2D):
	return image > random2D
