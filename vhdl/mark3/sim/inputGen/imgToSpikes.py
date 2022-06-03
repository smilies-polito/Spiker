import sys
from files import *
from parameters import *

if pythonSrcDir not in sys.path:
	sys.path.insert(1, pythonSrcDir)

from poisson import imgToSpikeTrain
from mnist import loadDataset

inputIndex = int(sys.argv[1])

# Load the MNIST dataset
imgArray, labelsArray = loadDataset(testImages, testLabels)

# Convert the image into spikes
spikesTrains = imgToSpikeTrain(imgArray[inputIndex], dt, trainingSteps, inputIntensity, rng)

with open(inSpikesFilename, "w") as spikes_fp:

	for inputSpikes in spikesTrains:
		spikes_fp.write(str(list(inputSpikes.astype(int))).replace(",",
			"").replace(" ", "")[1:-1])
		spikes_fp.write("\n")


