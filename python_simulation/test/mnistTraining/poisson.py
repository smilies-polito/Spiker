#!/Users/alessio/anaconda3/bin/python3

import numpy as np

def randInt3Dgen(N_images, timeEvolCycles, N_pixels, pixelMin, pixelMax):

	rng = np.random.default_rng()

	return rng.integers(low=pixelMin, high=pixelMax, size=(N_images, timeEvolCycles, 
		N_pixels))




def poisson(image, random2D):
	return image > random2D
