import numpy as np
import os
import h5py

import tonic
from tonic import datasets, transforms

import snntorch as snn
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_shd_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def sparse_data_generator_from_hdf5_spikes(X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
    """ This generator takes a spike dataset and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y,dtype=int)
    number_of_batches = len(labels_)//batch_size
    sample_index = np.arange(len(labels_))

    # compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']
    
    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]
            
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
    
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index],device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1

# Define Network
class Net(nn.Module):
	def __init__(self, num_inputs, num_hidden, num_outputs, alpha, beta,
			sigmoid_slope):

		super().__init__()

		#Network dimensions
		self.num_inputs = num_inputs
		self.num_hidden = num_hidden
		self.num_outputs = num_outputs

		# Fast sigmoid surrogate gradient
		self.spike_grad = surrogate.fast_sigmoid(slope =
				sigmoid_slope)

		# Initialize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden)
		self.fb1 = nn.Linear(num_hidden, num_hidden)

		self.lif1 = snn.Synaptic(alpha = alpha, beta = beta, 
				spike_grad = self.spike_grad)

		self.fc2 = nn.Linear(num_hidden, num_outputs)

		self.lif2 = snn.Synaptic(alpha = alpha, beta = beta, 
				spike_grad = self.spike_grad)

	def forward(self, input_spikes):

		# Initialize hidden states at t=0
		syn1, mem1 = self.lif1.init_synaptic()
		syn2, mem2 = self.lif2.init_synaptic()

		# Record the final layer
		spk2_rec = []
		mem2_rec = []

		input_spikes = input_spikes.float()
		spk1 = torch.zeros(self.num_hidden).to(device)


		for step in range(input_spikes.shape[1]):
			cur1 = self.fc1(input_spikes[:, step, :]) + \
				self.fb1(spk1)
			spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
			cur2 = self.fc2(spk1)
			spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

			spk2_rec.append(spk2)
			mem2_rec.append(mem2)

		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def print_batch_accuracy(net, data, batch_size, num_steps, targets, train=False):

	spk_rec, mem_rec= net(data.to_dense())
	
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

max_time = 1.4

sigmoid_slope	= 100

# Regularization
r1		= 2e-6
r2		= 2e-6

# Optimizer
adam_beta1	= 0.9
adam_beta2	= 0.999
lr		= 2e-4

# Here we load the Dataset
cache_dir = os.path.expanduser("~/data")
cache_subdir = "hdspikes"
get_shd_dataset(cache_dir, cache_subdir)

train_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_train.h5'), 'r')
test_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_test.h5'), 'r')

x_train = train_file['spikes']
y_train = train_file['labels']
x_test = test_file['spikes']
y_test = test_file['labels']


net = Net(
	num_inputs	= num_inputs,
	num_hidden	= num_hidden,
	num_outputs	= num_outputs,
	alpha		= alpha,
	beta		= beta,
	sigmoid_slope	= sigmoid_slope
)


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

	# Minibatch training loop
	for data, labels in sparse_data_generator_from_hdf5_spikes(x_train,
			y_train, batch_size, n_samples, num_inputs, max_time, shuffle=True):

		# Forward pass
		net.train()

		spk_rec, mem_rec = net(data.to_dense())
		m, _ = torch.max(mem_rec, 0)
		log_p_y = log_softmax_fn(m)

		# L1 loss on total number of spikes
		reg_loss = r1 * torch.sum(spk_rec)

		# L2 loss on spikes per neuron
		reg_loss += r2 * torch.mean(torch.sum(torch.sum(spk_rec, dim=0), dim=0)**2)

		loss_val = loss_fn(log_p_y, labels) + reg_loss

		# Gradient calculation + weight update
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		# Store loss history for future plotting
		loss_hist.append(loss_val.item())

		# Test set
		with torch.no_grad():

			net.eval()

			test_data, test_labels = next(sparse_data_generator_from_hdf5_spikes(x_test,
			y_test, batch_size, n_samples, num_inputs, max_time,
			shuffle=False))

			# Test set forward pass
			test_spk_rec, test_mem_rec = net(test_data.to_dense())

			test_m, _ = torch.max(test_mem_rec, 0)
			test_log_p_y = log_softmax_fn(test_m)

			test_loss = loss_fn(test_log_p_y, test_labels)

			# Test set loss
			test_loss_hist.append(test_loss.item())

			# Print train/test loss/accuracy
			if counter % 1 == 0 and counter > 0:
				train_printer(net, batch_size, n_samples, epoch,
						iter_counter, loss_hist,
						test_loss_hist, counter,
						data, labels,
						test_data,
						test_labels)

			counter += 1
			iter_counter +=1
