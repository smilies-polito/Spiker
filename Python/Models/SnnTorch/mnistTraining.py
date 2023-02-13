import torch
import torch.nn as nn
import numpy as np

from mnist import loadDataset
from network import Net
from trainTestFunctions import train_printer
from utils import createDir

from files import *
from runParameters import *

createDir(paramDir)

train_loader, test_loader = loadDataset(data_path, batch_size)

net = Net(num_inputs, num_hidden, num_outputs, beta)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))


# Outer training loop
for epoch in range(num_epochs):

	iter_counter = 0
	train_batch = iter(train_loader)

	# Minibatch training loop
	for data, targets in train_batch:

		# Forward pass
		net.train()
		spk_rec, mem_rec = net(data.view(batch_size, -1), num_steps)

		# Initialize the loss & sum over time
		loss_val = torch.zeros((1), dtype=dtype)
		for step in range(num_steps):
			loss_val += loss(mem_rec[step], targets)

		# Gradient calculation + weight update
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		# Store loss history for future plotting
		loss_hist.append(loss_val.item())

		# Test set
		with torch.no_grad():
			net.eval()
			test_data, test_targets = next(iter(test_loader))

			# Test set forward pass
			test_spk, test_mem = net(test_data.view(batch_size, -1),
					num_steps)

			# Test set loss
			test_loss = torch.zeros((1), dtype=dtype)
			for step in range(num_steps):
				test_loss += loss(test_mem[step], test_targets)
			test_loss_hist.append(test_loss.item())

			# Print train/test loss/accuracy
			if counter % 50 == 0:
				train_printer(net, batch_size, num_steps, epoch,
						iter_counter, loss_hist,
						test_loss_hist, counter, data,
						targets, test_data,
						test_targets)
			counter += 1
			iter_counter +=1



with open(weightsFilename + "1.npy", "wb") as fp:
	np.save(fp, net.fc1.weight.data.numpy())

with open(weightsFilename + "2.npy", "wb") as fp:
	np.save(fp, net.fc2.weight.data.numpy())

with open(weightsFilename + "1.pt", "wb") as fp:
	torch.save(net.fc1.weight.data, fp)

with open(weightsFilename + "2.pt", "wb") as fp:
	torch.save(net.fc2.weight.data, fp)
