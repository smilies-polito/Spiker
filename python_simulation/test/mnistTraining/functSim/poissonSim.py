#!/Users/alessio/anaconda3/bin/python3

import sys

previousDir = ".."

if previousDir not in sys.path:
	sys.path.insert(1, previousDir)

import numpy as np
import matplotlib.pyplot as plt


from poisson import poisson, randInt3Dgen

N_images = 1
timeEvolCycles = 100
N_pixels = 256
pixelMin = 0
pixelMax = 255

image = np.linspace(0, 255, 256)

random2D = randInt3Dgen(N_images, timeEvolCycles, N_pixels, pixelMin, pixelMax)[0]

imgSpikes = poisson(image, random2D)

imgSpikes = imgSpikes.T

for i in imgSpikes:
	plt.plot(i)
	plt.show()
