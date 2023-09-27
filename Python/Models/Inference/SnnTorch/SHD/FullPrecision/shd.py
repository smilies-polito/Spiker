import tonic

import numpy as np
import torch
from torch import nn

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

class SpikeFunction(torch.autograd.Function):

	scale = 100

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		out = torch.zeros_like(input)
		out[input>0] = 1.0
		return out


	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		return grad_output * 1/(1 +SpikeFunction.scale*
			torch.abs(input))**2


class LIF:

	def __init__(self, alpha, beta, n_neurons, threshold = 1.0, 
	readout = False):
		self.n_neurons	= n_neurons
		self.syn 	= torch.zeros(n_neurons)
		self.alpha	= alpha
		self.mem 	= torch.zeros(n_neurons)
		self.beta	= beta
		self.th		= threshold
		self.out	= torch.zeros(n_neurons)

		self.readout	= readout

		# Spike function with surrogate gradient
		self.spike_fn = SpikeFunction.apply

	def init_neurons(self):
		self.mem = torch.zeros(self.n_neurons)
		self.syn = torch.zeros(self.n_neurons)
		self.out = torch.zeros(self.n_neurons)

	def exp_decay(self, value, decay_rate):
		return value*decay_rate

	def reset(self, value):
		rst = self.out.detach()
		value[rst == 1] = 0

	def __call__(self, input_current):

		if not self.readout:
			self.reset(self.mem)

		mthr = self.mem - self.th

		self.out = self.spike_fn(mthr)

		self.mem = self.exp_decay(self.mem, self.beta) + self.syn
		self.syn = self.exp_decay(self.syn, self.alpha) + input_current

		return self.out, self.mem


class SNN(nn.Module):

	def __init__(self, num_inputs, num_hidden, num_output, alpha, beta,
			threshold = 1.0, alpha2 = None, beta2 = None):

		super().__init__()

		self.num_inputs = num_inputs
		self.num_hidden = num_hidden
		self.num_output = num_output

		if not alpha2:
			alpha2 = alpha

		if not beta2:
			beta2 = beta

		# Input fully connected layer
		self.fc_in = nn.Linear(num_inputs, num_hidden, bias = False)

		# Feedback fully connected layer
		self.fc_fb = nn.Linear(num_hidden, num_hidden, bias = False)

		# Hidden layer of LIF neuron
		self.lif1 = LIF(alpha, beta, num_hidden, threshold)

		# Readout fully connected layer
		self.fc_ro = nn.Linear(num_hidden, num_output, bias = False)

		# Readout layer
		self.readout = LIF(alpha2, beta2, num_output, readout = True)

	def forward(self, input_spikes, n_steps = None):

		if not n_steps:
			n_steps = input_spikes.shape[0]
		
		elif n_steps != input_spikes.shape[0]:
			raise ValueError("Incompatible number of steps for "
			"input")

		hidden_spikes = torch.zeros(self.num_hidden)
		ro_mem = torch.zeros(self.num_output)

		ro_mem_rec = []

		self.lif1.init_neurons()
		self.readout.init_neurons()

		for i in range(n_steps):

			in_syn_curr = self.fc_in(input_spikes[i])
			fb_syn_curr = self.fc_fb(hidden_spikes)
			syn_curr = in_syn_curr + fb_syn_curr

			#print("----------------%d----------------\n"%(i))
			#print(syn_curr)
			#print("\n\n")

			hidden_spikes, _ = self.lif1(syn_curr)

			#print(hidden_spikes)
			#print("\n\n")
			#time.sleep(20)

			ro_syn_curr = self.fc_ro(hidden_spikes)
			_, ro_mem = self.readout(ro_syn_curr)

			ro_mem_rec.append(ro_mem)

		return 	torch.stack(ro_mem_rec, dim=1) \


def compute_classification_accuracy(dataset):

	""" 
	Computes classification accuracy on supplied data in batches.
	"""

	accs = []

	for data in dataset:

		dense_data = transform(data[0]).to_dense()
		label = data[1]

		output = snn(dense_data)

		# Max over time
		m,_= torch.max(output, 1)

		# Argmax over output units
		_, am=torch.max(m, 0)

		accs.append(label == am)

	return np.mean(accs)



# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

dtype = torch.float

batch_size = 1
n_samples = 100
n_inputs = tonic.datasets.hsd.SHD.sensor_size[0]
n_hidden = 200
n_output = 20
min_time = 0
max_time = 1.4 * 10**6

time_step = 1e-3
tau_mem = 10e-3
tau_syn = 5e-3

alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))

dataset = tonic.datasets.hsd.SHD(save_to='./data', train=False)

transform = TonicTransform(
	min_time	= min_time,
	max_time	= max_time,
	n_samples	= n_samples,
	n_inputs	= n_inputs
)

snn = SNN(
	num_inputs = n_inputs,
	num_hidden = n_hidden,
	num_output = n_output,
	alpha = alpha,
	beta = beta,
)

snn.fc_in.weight.data = torch.load("w1.pt",
		map_location=torch.device('cpu')).transpose(0, 1)
snn.fc_fb.weight.data = torch.load("v1.pt",
		map_location=torch.device('cpu')).transpose(0, 1)
snn.fc_ro.weight.data = torch.load("w2.pt",
		map_location=torch.device('cpu')).transpose(0, 1)

print(compute_classification_accuracy(dataset))
