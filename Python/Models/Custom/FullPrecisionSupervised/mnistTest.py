import torch
import torch.nn as nn
import numpy as np

from snntorch import spikegen

from createNetwork import createNetwork
from network import run
from trainTestFunctions import train_printer, test_printer
from utils import createDir

from files import *
from runParameters import *

mode = "test"

# Create the network data structure
net = createNetwork(networkList, weightsFilename, thresholdsFilename, mode, 
			excDictList, scaleFactors, inh2excWeights)

acc = 0

with open(inputSpikes, "rb") as fp:
	spikesTrainsBatch = torch.load(fp)



with open(logFile, "w") as fp:

	for i in range(spikesTrainsBatch.size()[1]):

		spikesTrains = spikesTrainsBatch.numpy().astype(bool)[:, i, :]

		outputCounters = run(net, networkList, spikesTrains, dt_tauDict,
				None, mode, None)

		outputLabel = np.where(outputCounters[0] ==
				np.max(outputCounters[0]))[0][0]

		for counter in outputCounters[0]:
			fp.write(str(int(counter)))
			fp.write("\t")
		fp.write("\t\t")
		fp.write(str(outputLabel))
		fp.write("\n")
