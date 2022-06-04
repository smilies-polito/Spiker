import subprocess as sp
import numpy as np

from createNetwork import createNetwork
from parameters import *
from runParameters import *

with open("../inputOutput/outWeights.txt") as fp:
	
	vhdlWeights = np.zeros((784, 400)).astype(int)
	samples = np.zeros(784).astype(int)
	exc_cnt = np.zeros(784).astype(int)

	i=0

	for line in fp:

		sample, exc_cnt_string, weights_string = line.split()

		samples[i] = int(sample)

		exc_cnt[i] = int(exc_cnt_string, 2)

		for j in range(400):

			vhdlWeights[i][j] = int(weights_string[j*5:(j+1)*5], 2)

		i += 1




# Create the network data structure
network = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, trainPrecision, rng)

pythonWeights = network["exc2exc1"]["weights"]

with open("../vhdlTmp.txt", "w") as fp:
	fp.write(str(list(vhdlWeights.T[0].astype(int))).replace(",",
		"").replace(" ", "\n")[1:-1])
