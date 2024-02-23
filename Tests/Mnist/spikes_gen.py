
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

import snntorch as snn
from snntorch import spikegen

np.set_printoptions(threshold = np.inf)

class Net(nn.Module):
	def __init__(self, num_inputs, num_hidden, num_outputs, beta):
		super().__init__()

		# Initialize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden)
		self.lif1 = snn.Leaky(beta=beta)
		self.fc2 = nn.Linear(num_hidden, num_outputs)
		self.lif2 = snn.Leaky(beta=beta)

	def forward(self, data_it, num_steps):

		# Initialize hidden states at t=0
		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()

		# Record the final layer
		spk2_rec = []
		mem2_rec = []

		input_spikes = spikegen.rate(data_it, num_steps = num_steps,
				gain = 1)

		for step in range(num_steps):
			cur1 = self.fc1(input_spikes[step])
			spk1, mem1 = self.lif1(cur1, mem1)
			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)
			spk2_rec.append(spk2)
			mem2_rec.append(mem2)

		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)




data_dir 	= './data'

target_dir 	= "./VhdlSpiker"
spikes_file 	= target_dir + "/in_spikes.txt"
out_file	= target_dir + "/out_spikes_python.txt"

image_height	= 28
image_width	= 28
num_steps	= 100

num_inputs	= image_height*image_width
num_hidden	= 128
num_outputs	= 10

beta 		= 0.9375

image_index	= 13

transform = transforms.Compose([
	transforms.Resize((image_width, image_height)),
	transforms.Grayscale(),
	transforms.ToTensor(),
	transforms.Normalize((0,), (1,))]
)

test_set = datasets.MNIST(root=data_dir, train=False, download=True,
		transform=transform)

test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
		drop_last=True)

test_data = list(iter(test_loader))

image = test_data[image_index][0].view(-1) 

spikes = spikegen.rate(image, num_steps = num_steps, gain = 1)

with open(spikes_file, "w") as fp:
	for t in range(spikes.shape[0]):
		s = spikes[t].int().tolist()
		s = str(s)[1:-1].replace(" ", "").replace("\n", "").replace(",",
				"")
		fp.write(s)
		fp.write("\n")

net = Net(
	num_inputs	= num_inputs,
	num_hidden	= num_hidden,
	num_outputs	= num_outputs,
	beta		= beta
)

spk_rec, _ = net(test_data[image_index][0].view(1, -1), num_steps)

spk_rec = spk_rec.view(num_steps, -1)

with open(out_file, "w") as fp:
	for t in range(spk_rec.shape[0]):
		s = spk_rec[t].int().tolist()
		s = str(s)[1:-1].replace(" ", "").replace("\n", "").replace(",",
				"")
		fp.write(s)
		fp.write("\n")
