import torch
import torch.nn as nn
import numpy as np

from snntorch import spikegen

from mnist import loadDataset
from createNetwork import createNetwork
from network import run
from trainTestFunctions import train_printer, test_printer
from utils import createDir

from files import *
from runParameters import *

mode = "test"

train_loader, test_loader = loadDataset(data_path, batch_size)

# Create the network data structure
net = createNetwork(networkList, weightsFilename, thresholdsFilename, mode, 
			excDictList, scaleFactors, inh2excWeights)

test_batch = iter(test_loader)

# Minibatch training loop
for test_data, test_targets in test_batch:

	acc = 0

	for i in range(test_data.size()[0]):

		image = test_data[i, :, :, :]
		label = int(test_targets[i].int())

		spikesTrains = spikegen.rate(image.reshape(networkList[0], 1),
				num_steps).numpy().astype(bool)[:, :, 0]

		outputCounters = run(net, networkList, spikesTrains, dt_tauDict,
				None, mode, None)
		print(outputCounters)

		outputLabel = np.where(np.max(outputCounters))[0][0]

		if outputLabel == label:
			acc += 1

	
	acc = acc / test_data.size()[0]
	print(f"Accuracy: {acc*100}%")

	
