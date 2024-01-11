import numpy as np
import tonic

import snntorch as snn
from snntorch import spikegen

import torch
import torch.nn as nn

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

		spk1 = torch.zeros(self.num_hidden)

		for step in range(input_spikes.shape[0]):
			cur1 = self.fc1(input_spikes[step]) + self.fb1(spk1)
			spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
			cur2 = self.fc2(spk1)
			spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

			spk2_rec.append(spk2)
			mem2_rec.append(mem2)

		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

class TonicTransform:

	def __init__(self, min_time : float, max_time : float, n_samples :
			int = 100, n_inputs : int = 700):

		self.min_time	= min_time
		self.max_time	= max_time
		self.n_samples	= n_samples
		self.n_inputs	= n_inputs

	def __call__(self, sparse_tensor : torch.Tensor) -> torch.Tensor:
		return self.events_to_sparse(sparse_tensor)


	def events_to_sparse(self, sparse_tensor : torch.Tensor) -> \
		torch.Tensor:

		assert "t" and "x" 
		times = self.resample(sparse_tensor["t"])

		units = sparse_tensor["x"]

		indexes = np.stack((times, units), axis = 0)
		values = np.ones(times.shape[0])

		return torch.sparse_coo_tensor(indexes, values, (self.n_samples,
			self.n_inputs), dtype = torch.float)

	def resample(self, np_array : np.array) -> np.array:

		sampling_index = np.linspace(
			self.min_time,
			self.max_time,
			self.n_samples
		)

		return np.digitize(np_array, sampling_index)


def compute_classification_accuracy(net, data_label_couples):

	""" 
	Computes classification accuracy on supplied data in batches.
	"""

	accs = []

	for sample in data_label_couples:

		dense_sample = transform(sample[0]).to_dense()
		label = sample[1]

		_, mem_rec = net(dense_sample)

		# Max over time
		m,_= torch.max(mem_rec, 1)

		# Argmax over output units
		_, am=torch.max(m, 0)

		accs.append(label == am)

	return np.mean(accs)


def train_printer(net, batch_size, epoch, iter_counter, loss_hist,
		test_loss_hist, counter, train_couples, test_couples):

	print(f"Epoch {epoch}, Iteration {iter_counter}")

	print(f"Train Set Loss: {loss_hist[counter]:.2f}")

	print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")

	print(compute_classification_accuracy(net, train_couples))

	print(compute_classification_accuracy(net, test_couples))


	print("\n")


n_epochs	= 300


n_samples	= 100
num_inputs	= tonic.datasets.hsd.SHD.sensor_size[0]
num_hidden	= 200
num_outputs	= 20

time_step	= 1e-3
batch_size	= 1

tau_mem		= 10e-3
tau_syn		= 5e-3

alpha		= float(np.exp(-time_step/tau_syn))
beta		= float(np.exp(-time_step/tau_mem))

min_time	= 0
max_time	= 1.4 * 10**6

train_set 	= tonic.datasets.hsd.SHD(save_to='./data', train=True)
test_set 	= tonic.datasets.hsd.SHD(save_to='./data', train=False)

transform = TonicTransform(
	min_time	= min_time,
	max_time	= max_time,
	n_samples	= n_samples,
	n_inputs	= num_inputs
)

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

net = Net(num_inputs, num_hidden, num_outputs, alpha, beta)

log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

loss_hist = []
test_loss_hist = []
train_couples = []
test_couples = []
counter = 0

# Outer training loop
for epoch in range(n_epochs):

	iter_counter = 0

	# Minibatch training loop
	for train_data in train_set:

		# Forward pass
		net.train()

		dense_data = transform(train_data[0]).to_dense()
		label = torch.tensor(train_data[1])
		label = torch.reshape(label, (-1,))
		
		spk_rec, mem_rec = net(dense_data)
		m, _ = torch.max(mem_rec, 0)
		m = torch.reshape(m, (1, -1))
		log_p_y = log_softmax_fn(m)

		loss_val = loss_fn(log_p_y, label)

		# Gradient calculation + weight update
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		# Store loss history for future plotting
		loss_hist.append(loss_val.item())

		# Test set
		with torch.no_grad():
			net.eval()

			test_couple = next(iter(test_set))
			test_data = transform(test_couple[0]).to_dense()
			test_label = torch.tensor(test_couple[1])
			test_label = torch.reshape(test_label, (-1,))

			# Test set forward pass
			test_spk_rec, test_mem_rec = net(test_data)

			test_m, _ = torch.max(test_mem_rec, 0)
			test_m = torch.reshape(test_m, (1, -1))
			test_log_p_y = log_softmax_fn(test_m)

			test_loss = loss_fn(test_log_p_y, test_label)

			# Test set loss
			test_loss_hist.append(test_loss.item())

			# Print train/test loss/accuracy
			if counter % 50 == 0 and counter > 0:
				train_printer(net, batch_size, epoch,
						iter_counter, loss_hist,
						test_loss_hist, counter,
						train_couples, test_couples)

				train_couples = []
				test_couples = []

			else:
				train_couples.append(train_data) 
				test_couples.append(test_couple)

			counter += 1
			iter_counter +=1
