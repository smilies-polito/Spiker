import numpy as np

import snntorch as snn
from snntorch import spikegen

import torch
import torch.nn as nn

from files import *

# Define Network
class Net(nn.Module):
	def __init__(self, num_inputs, num_hidden, num_outputs, beta):
		super().__init__()

		# Initialize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden, bias = False)
		self.lif1 = snn.Leaky(beta=beta)
		self.fc2 = nn.Linear(num_hidden, num_outputs, bias = False)
		self.lif2 = snn.Leaky(beta=beta)

	def forward(self, data_it, num_steps):

		# Initialize hidden states at t=0
		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()

		# Record the final layer
		spk2_rec = []
		mem2_rec = []

		with open(inputSpikes, "rb") as fp:
			input_spikes = torch.load(fp)

		for step in range(num_steps):
			cur1 = self.fc1(input_spikes[step])
			print(float(cur1[step, 0]))
			spk1, mem1 = self.lif1(cur1, mem1)
			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)
			spk2_rec.append(spk2)
			mem2_rec.append(mem2)

		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

