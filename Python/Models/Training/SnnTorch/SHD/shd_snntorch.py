import numpy as np

import tonic
from tonic import datasets, transforms

import snntorch as snn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Define Network
class Net(nn.Module):
	def __init__(self, num_inputs, num_hidden, num_outputs, alpha, beta):
		super().__init__()

		self.num_inputs = num_inputs
		self.num_hidden = num_hidden
		self.num_outputs = num_outputs

		# Initialize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden)
		self.fb1 = nn.Linear(num_hidden, num_hidden)
		self.lif1 = snn.Synaptic(alpha = alpha, beta = beta)
		self.fc2 = nn.Linear(num_hidden, num_outputs)
		self.lif2 = snn.Synaptic(alpha = alpha, beta = beta)

	def forward(self, input_spikes):

		# Initialize hidden states at t=0
		syn1, mem1 = self.lif1.init_synaptic()
		syn2, mem2 = self.lif2.init_synaptic()

		# Record the final layer
		spk2_rec = []
		mem2_rec = []

		spk1 = torch.zeros(self.num_hidden).to(device)
		input_spikes = input_spikes.float()

		for step in range(input_spikes.shape[1]):
			cur1 = self.fc1(input_spikes[:, step, :]) + self.fb1(spk1)
			spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
			cur2 = self.fc2(spk1)
			spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

			spk2_rec.append(spk2)
			mem2_rec.append(mem2)

		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def print_batch_accuracy(net, data, batch_size, num_steps, targets, train=False):

	spk_rec, mem_rec= net(data[:, :, 0, :])
	
	m, _ = torch.max(mem_rec, 0) # max over time

	_, am = torch.max(m, 1)      # argmax over output units

	acc = np.mean((targets == am).detach().cpu().numpy())

	if train:
		print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
	else:
		print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def train_printer(net, batch_size, num_steps, epoch, iter_counter, loss_hist,
		test_loss_hist, counter, data, targets, test_data,
		test_targets):

	print(f"Epoch {epoch}, Iteration {iter_counter}")

	print(f"Train Set Loss: {loss_hist[counter]:.2f}")

	print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")

	print_batch_accuracy(net, data, batch_size, num_steps, targets, train=True)

	print_batch_accuracy(net, test_data, batch_size, num_steps, test_targets, 
			train=False)

	print("\n")


n_epochs	= 200
n_samples	= 100
num_inputs	= tonic.datasets.hsd.SHD.sensor_size[0]
num_hidden	= 200
num_outputs	= 20

time_step	= 1e-3
batch_size	= 256

tau_mem		= 10e-3
tau_syn		= 5e-3

alpha		= float(np.exp(-time_step/tau_syn))
beta		= float(np.exp(-time_step/tau_mem))

min_time	= 0
max_time	= 1.4 * 10**6

transform = transforms.Compose(
    [
        transforms.ToFrame(
            sensor_size=tonic.datasets.hsd.SHD.sensor_size,
            n_time_bins=100,
        )
    ]
)

train_set 	= tonic.datasets.hsd.SHD(save_to='./data', train=True, transform
		= transform)
test_set 	= tonic.datasets.hsd.SHD(save_to='./data', train=False,
		transform = transform)

train_loader 	= DataLoader(train_set, batch_size=batch_size, shuffle=True,
		drop_last=True)
test_loader 	= DataLoader(test_set, batch_size=batch_size, shuffle=True,
		drop_last=True)

net = Net(num_inputs, num_hidden, num_outputs, alpha, beta)


log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

loss_hist = []
test_loss_hist = []
train_couples = []
test_couples = []
counter = 0

net.to(device)

# Outer training loop
for epoch in range(n_epochs):

	iter_counter = 0
	train_batches	= iter(train_loader)

	# Minibatch training loop
	for data, labels in train_batches:

		inputs = data.to(device)
		y = labels.to(device)

		# Forward pass
		net.train()

		spk_rec, mem_rec = net(inputs[:, :, 0, :])
		m, _ = torch.max(mem_rec, 0)
		log_p_y = log_softmax_fn(m)

		loss_val = loss_fn(log_p_y, y)

		# Gradient calculation + weight update
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		# Store loss history for future plotting
		loss_hist.append(loss_val.item())

		# Test set
		with torch.no_grad():

			net.eval()

			test_data, test_labels = next(iter(test_loader))

			test_inputs = test_data.to(device)
			test_y = test_labels.to(device)

			# Test set forward pass
			test_spk_rec, test_mem_rec = net(test_inputs[:, :, 0, :])

			test_m, _ = torch.max(test_mem_rec, 0)
			test_log_p_y = log_softmax_fn(test_m)

			test_loss = loss_fn(test_log_p_y, test_y)

			# Test set loss
			test_loss_hist.append(test_loss.item())

			# Print train/test loss/accuracy
			if counter % 50 == 0 and counter > 0:
				train_printer(net, batch_size, n_samples, epoch,
						iter_counter, loss_hist,
						test_loss_hist, counter, inputs,
						y, test_inputs,
						test_y)

			counter += 1
			iter_counter +=1
