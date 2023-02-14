import torch
import torch.nn as nn
import numpy as np

from mnist import loadDataset
from network import Net
from trainTestFunctions import train_printer, test_printer
from utils import createDir

from files import *
from runParameters import *


net = Net(num_inputs, num_hidden, num_outputs, beta)

net.fc1.weight.data = torch.load(weightsFilename + "1.pt")
net.fc2.weight.data = torch.load(weightsFilename + "2.pt")

loss = nn.CrossEntropyLoss()

with open(logFile, "w") as fp:

	# Test set
	with torch.no_grad():
		net.eval()

		# Test set forward pass
		test_spk, test_mem = net(None, # test_data.view(batch_size, -1),
				num_steps)

		outputCounters = test_spk.sum(dim=0)

		outputLabel = outputCounters.max(1)

	for i in range(outputCounters.size()[0]):
		for counter in outputCounters[i]:
			fp.write(str(int(counter)))
			fp.write("\t")
		fp.write("\t")
		fp.write(str(int(outputLabel[1][i])))
		fp.write("\n")
