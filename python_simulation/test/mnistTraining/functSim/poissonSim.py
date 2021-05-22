#!/Users/alessio/anaconda3/bin/python3

import sys

previousDir = ".."

if previousDir not in sys.path:
	sys.path.insert(1, previousDir)

import numpy as np
import matplotlib.pyplot as plt


from poisson import imgToSpikeTrain

timeEvolCycles = 100
N_pixels = 256
pixelMin = 0
pixelMax = 255

image = np.linspace(0, 255, 256)

imgSpikes = imgToSpikeTrain(image, timeEvolCycles, N_pixels, pixelMin, pixelMax)

imgSpikes = imgSpikes.T

for i in imgSpikes:
	plt.plot(i)
	plt.show()
