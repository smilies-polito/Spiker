import snntorch as snn

import torch
import torch.nn as nn

# Define Network
class Net(nn.Module):
	def __init__(self, num_inputs, num_hidden, num_outputs, beta):
		super().__init__()

		# Initialize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden)
		self.lif1 = snn.Leaky(beta=beta)
		self.fc2 = nn.Linear(num_hidden, num_outputs)
		self.lif2 = snn.Leaky(beta=beta)

	def forward(self, input_spikes, num_steps):

		# Initialize hidden states at t=0
		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()

		# Record the final layer
		spk2_rec = []
		mem2_rec = []

		for step in range(num_steps):
			cur1 = self.fc1(input_spikes)
			spk1, mem1 = self.lif1(cur1, mem1)
			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)
			spk2_rec.append(spk2)
			mem2_rec.append(mem2)

		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

