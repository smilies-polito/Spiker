import snntorch as snn

import torch
import torch.nn as nn

# Define Network
class Net(nn.Module):
	def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs,
			beta):
		super().__init__()

		# Initialize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden1)
		self.lif1 = snn.Leaky(beta=beta)
		self.fc2 = nn.Linear(num_hidden1, num_hidden2)
		self.lif2 = snn.Leaky(beta=beta)
		self.fc3 = nn.Linear(num_hidden2, num_outputs)
		self.lif3 = snn.Leaky(beta=beta)

	def forward(self, input_spikes, num_steps):

		# Initialize hidden states at t=0
		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()
		mem3 = self.lif3.init_leaky()

		# Record the final layer
		spk3_rec = []
		mem3_rec = []

		for step in range(num_steps):
			cur1 = self.fc1(input_spikes)
			spk1, mem1 = self.lif1(cur1, mem1)
			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)
			cur3 = self.fc3(spk2)
			spk3, mem3 = self.lif3(cur3, mem3)
			spk3_rec.append(spk3)
			mem3_rec.append(mem3)

		return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

