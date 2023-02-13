import torch
import torch.nn as nn
import numpy as np

from mnist import loadDataset
from network import Net
from trainTestFunctions import train_printer, test_printer
from utils import createDir

from files import *
from runParameters import *

train_loader, test_loader = loadDataset(data_path, batch_size)

net = Net(num_inputs, num_hidden, num_outputs, beta)

net.fc1.weight.data = torch.load(weightsFilename + "1.pt")
net.fc2.weight.data = torch.load(weightsFilename + "2.pt")

loss = nn.CrossEntropyLoss()

test_batch = iter(test_loader)

print(len(test_batch))
iter_counter = 0

# Minibatch training loop
for test_data, test_targets in test_batch:

	# Test set
	with torch.no_grad():
		net.eval()

		# Test set forward pass
		test_spk, test_mem = net(test_data.view(batch_size, -1),
				num_steps)

		# Test set loss
		test_loss = torch.zeros((1), dtype=dtype)
		for step in range(num_steps):
			test_loss += loss(test_mem[step], test_targets)
		test_loss_hist.append(test_loss.item())

		# Print train/test loss/accuracy
		test_printer(net, batch_size, num_steps, iter_counter,
				test_loss_hist, counter, test_data,
				test_targets)
		counter += 1
		iter_counter +=1
