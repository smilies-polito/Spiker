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

# Create a fictitious image with all the possible values that the pixels can assume
image = np.linspace(0, 255, 256)

# Convert the image into spike trains
imgSpikes = imgToSpikeTrain(image, timeEvolCycles, N_pixels, pixelMin, pixelMax)

# Transpose the array in order to have one element for each pixel, each element containing
# the temporal evolution of the spikes associated to the pixel.
imgSpikes = imgSpikes.T


# Plot the spike train associated to each pixel
for i in imgSpikes:
	plt.plot(i)
	plt.show()
