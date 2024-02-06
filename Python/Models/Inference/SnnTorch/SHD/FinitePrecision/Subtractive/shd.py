import tonic

import numpy as np
import torch
from torch import nn
from functools import partial

class Quantize:

	def saturate(self, value, bitwidth):

		if type(value).__module__ == np.__name__ or \
		type(value).__module__ == torch.__name__:

			value[value > 2**(bitwidth-1)-1] = \
				2**(bitwidth-1)-1
			value[value < -2**(bitwidth-1)] = \
				-2**(bitwidth-1)
		else:
			if value > 2**(bitwidth-1)-1:
				value = 2**(bitwidth-1)-1

			elif value < -2**(bitwidth-1):
				value = -2**(bitwidth-1)

		return value

	def to_int(self, value):

		if type(value).__module__ == np.__name__:
			quant = value.astype(int)

		elif type(value).__module__ == torch.__name__:
			quant = value.type(torch.int64)

		else:
			quant = int(value)

		return quant

	def saturated_int(self, value, bitwidth):
		return self.saturate(self.to_int(value), bitwidth)

	def fixed_point(self, value, fp_dec, bitwidth):

		quant = value * 2**fp_dec

		return self.to_int(self.saturate(quant, bitwidth))


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


class LIF:

	def __init__(self, alpha_shift, beta_shift, n_neurons, threshold = 1.0, 
	readout = False):

		self.quant		= Quantize()

		self.n_neurons		= n_neurons
		self.syn		= self.quant.to_int(torch.zeros(
						n_neurons))
		self.alpha_shift	= alpha_shift
		self.mem		= self.quant.to_int(torch.zeros(
						n_neurons))
		self.beta_shift		= beta_shift
		self.th			= threshold
		self.out		= self.quant.to_int(torch.zeros(
						n_neurons))
		self.readout		= readout

	def init_neurons(self):
		self.mem = self.quant.to_int(torch.zeros(self.n_neurons))
		self.syn = self.quant.to_int(torch.zeros(self.n_neurons))
		self.out = self.quant.to_int(torch.zeros(self.n_neurons))

	def exp_decay(self, value, decay_shift):
		return self.quant.to_int(value - value*2**(-decay_shift))

	def reset(self, value):
		rst = self.out.detach()
		value[rst == 1] = value[rst == 1] - self.th

	def spike_fn(self, mthr):
		self.out = torch.zeros_like(mthr)
		self.out[mthr > 0] = 1


	def __call__(self, input_current):

		if not self.readout:
			self.reset(self.mem)

		mthr = self.quant.to_int(self.mem - self.th)

		self.spike_fn(mthr)

		self.mem = self.quant.to_int(self.exp_decay(self.mem,
			self.beta_shift) + self.quant.to_int(self.syn))

		self.syn = self.quant.to_int(self.exp_decay(self.syn,
			self.alpha_shift) + self.quant.to_int(input_current))

		return self.quant.to_int(self.out), self.quant.to_int(self.mem)


class SNN(nn.Module):

	def __init__(self, num_inputs, num_hidden, num_output, alpha_shift,
			beta_shift, w1, v1, w2, bitwidth1 = 64, bitwidth2 = 64,
			threshold = 1.0, alpha_shift2 = None, beta_shift2 =
			None):

		super().__init__()

		self.num_inputs = num_inputs
		self.num_hidden = num_hidden
		self.num_output = num_output

		self.bitwidth1 = bitwidth1
		self.bitwidth2 = bitwidth2

		self.w1 = w1
		self.v1 = v1
		self.w2 = w2

		print(self.w1)
		print(self.v1)
		print(self.w2)

		if not alpha_shift2:
			alpha_shift2 = alpha_shift

		if not beta_shift2:
			beta_shift2 = beta_shift

		if type(alpha_shift) != int or type(beta_shift) != int or \
		type(alpha_shift2) !=  int or type(beta_shift2) != int:
			raise ValueError("Invalid exponential shift value")

		self.quant = Quantize()

		# Input fully connected layer
		self.fc_in = partial(torch.einsum, "ab, b -> a")

		# Feedback fully connected layer
		self.fc_fb = partial(torch.einsum, "ab, b -> a")

		# Hidden layer of LIF neuron
		self.lif1 = LIF(alpha_shift, beta_shift, num_hidden, threshold)

		# Readout fully connected layer
		self.fc_ro = partial(torch.einsum, "ab, b -> a")

		# Readout layer
		self.readout = LIF(alpha_shift2, beta_shift2, num_output,
				readout = True)

	def forward(self, input_spikes, n_steps = None):

		if not n_steps:
			n_steps = input_spikes.shape[0]
		
		elif n_steps != input_spikes.shape[0]:
			raise ValueError("Incompatible number of steps for "
			"input")

		hidden_spikes = self.quant.to_int(torch.zeros(self.num_hidden))
		ro_mem = self.quant.to_int(torch.zeros(self.num_output))

		ro_mem_rec = []

		self.lif1.init_neurons()
		self.readout.init_neurons()

		for i in range(n_steps):

			in_syn_curr = self.quant.saturated_int(
					self.fc_in((self.w1.float(),
						input_spikes[i].float())),
					self.bitwidth1)
			fb_syn_curr = self.quant.saturated_int(
					self.fc_fb((self.v1.float(),
						hidden_spikes.float())),
					self.bitwidth1)
			syn_curr = self.quant.saturated_int(in_syn_curr +
					fb_syn_curr, self.bitwidth1)

			hidden_spikes, _ = self.lif1(syn_curr)

			ro_syn_curr = self.quant.saturated_int(self.fc_ro((
				self.w2.float(), hidden_spikes.float())), self.bitwidth2)

			_, ro_mem = self.readout(ro_syn_curr)

			ro_mem_rec.append(ro_mem)

		return 	torch.stack(ro_mem_rec, dim=1) \

def compute_classification_accuracy(dataset):

	""" 
	Computes classification accuracy on supplied data.
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



quantize = Quantize()

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

dtype = torch.float

# Network architecture
n_samples = 100
n_inputs = tonic.datasets.hsd.SHD.sensor_size[0]
n_hidden = 200
n_output = 20

# Conversion parameters
min_time = 0
max_time = 1.4 * 10**6

# Exponential shifts
alpha_shift = 4
beta_shift = 3

# Membrane threshold
threshold = quantize.fixed_point(1, fp_dec, bitwidth1)

# Import and transform data
dataset = tonic.datasets.hsd.SHD(save_to='./data', train=False)
transform = TonicTransform(
	min_time	= min_time,
	max_time	= max_time,
	n_samples	= n_samples,
	n_inputs	= n_inputs
)

# Bit-widths
fp_dec = 10
bitwidth1 = 64
bitwidth2 = 64
w_bitwidth1 = 64
w_bitwidth_fb1 = 64
w_bitwidth2 = 64

# Import trained weights
w1 = quantize.fixed_point(torch.load("w1.pt",
		map_location=torch.device('cpu')).transpose(0, 1), fp_dec,
		w_bitwidth1)
v1 = quantize.fixed_point(torch.load("v1.pt",
		map_location=torch.device('cpu')).transpose(0, 1), fp_dec,
		w_bitwidth_fb1)
w2 = quantize.fixed_point(torch.load("w2.pt",
		map_location=torch.device('cpu')).transpose(0, 1), fp_dec,
		w_bitwidth2)


snn = SNN(
	num_inputs	= n_inputs,
	num_hidden	= n_hidden,
	num_output	= n_output,
	threshold	= threshold,
	alpha_shift	= alpha_shift,
	beta_shift	= beta_shift,
	w1		= w1,
	v1		= v1,
	w2		= w2,
	bitwidth1	= bitwidth1,
	bitwidth2	= bitwidth2
)

print(compute_classification_accuracy(dataset))
