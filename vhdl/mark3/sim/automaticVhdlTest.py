import sys
import subprocess as sp

srcDir = "../../../python/custom/mark3_simplified"

if srcDir not in sys.path:
	sys.path.insert(1, srcDir)

from mnist import loadDataset
from poisson import imgToSpikeTrain

from runParameters import *

testSteps = int(trainDuration/dt)

N_out = 16
N_neurons = 400

testImages = "../../../../mnist/t10k-images-idx3-ubyte"
testLabels = "../../../../mnist/t10k-labels-idx1-ubyte"

inputsFilename = "inputs.txt"
outputsFilename = "cntOut.txt"


sp.run("make clean", shell=True)

sp.run("./compile.sh")

# Load the MNIST dataset
imgArray, labelsArray = loadDataset(testImages, testLabels)

for image in imgArray:

	# Convert the image into spikes trains
	spikesTrains = imgToSpikeTrain(image, dt, testSteps, inputIntensity,
			rng)

	with open(inputsFilename, "w") as fp:

		for step in spikesTrains:

			inputs = np.array2string(step.astype(int),
					max_line_width = step.shape[0]*2+2)

			inputs = inputs[1:-1]

			inputs = inputs.replace(" ", "")

			fp.write(inputs)
			fp.write("\n")


	sp.run("./sim.sh")

	with open(outputsFilename) as fp:

		countersString = fp.readline()

		counters = [0 for _ in range(N_neurons)]

		for i in range(N_neurons):

			binaryString = countersString[i*N_out:(i+1)*N_out]

			counters[i] = int(binaryString, 2)

	print(counters)

	break


sp.run("make clean", shell=True)
