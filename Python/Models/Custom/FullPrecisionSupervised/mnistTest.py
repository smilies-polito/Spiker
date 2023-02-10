import torch
import torch.nn as nn
import numpy as np

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

# # Minibatch training loop
# for test_data, test_targets in test_batch:
# 
# 	# Print train/test loss/accuracy
# 	test_printer(net, batch_size, num_steps, iter_counter,
# 				test_loss_hist, counter, test_data,
# 				test_targets)
