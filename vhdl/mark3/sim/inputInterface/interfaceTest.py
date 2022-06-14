import subprocess as sp
import numpy as np

from mnist import loadDataset
from poisson import imgToSpikeTrain
from runParameters import *
from files import *

# Load the MNIST dataset
imgArray , labelsArray = loadDataset(testImages, testLabels)

binaryFormat = "{0:0" + str(bitWidth) + "b}"

completeString = ""

for pixel in imgArray[0]:

	completeString += binaryFormat.format(pixel)
	
with open(vhdlInput, "w") as fp:
	fp.write(completeString)


sp.run(vhdlAutoTestbench)

vhdlSpikes = np.zeros((3500, 784)).astype(bool)


with open(vhdlOutput, "r") as fp:
	for i in range(3500):
		vhdlSpikes[i] = np.array(list(fp.readline()[:-1]))
		
spikes = imgToSpikeTrain(imgArray[0], dt, trainingSteps, inputIntensity, bitWidth, taps,
		seed)

with open(pythonOutput, "w") as fp:
	for step in spikes:
		fp.write(str(list(step.astype(int)))[1:-1].replace(",", "")
				.replace(" ", ""))
		fp.write("\n")


if (vhdlSpikes == spikes).all():
	print("\n\nCorrect\n\n")
else:
	print("\n\nWrong\n\n")
