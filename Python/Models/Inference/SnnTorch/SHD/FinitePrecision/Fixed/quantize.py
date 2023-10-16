import subprocess as sp
import pickle

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

	def to_float(self, value):

		if type(value).__module__ == np.__name__:
			quant = value.astype(float)

		elif type(value).__module__ == torch.__name__:
			quant = value.type(torch.float)

		else:
			quant = float(value)

		return quant

	def saturated_int(self, value, bitwidth):
		return self.to_int(self.saturate(value, bitwidth))

	def fixed_point(self, value, fp_dec, bitwidth):

		quant = self.to_float(value) * float(2**fp_dec)

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
		return self.quant.to_int(value) - \
			(self.quant.to_int(value) >> decay_shift)

	def reset(self, value):
		rst = self.out.detach()
		#value[rst == 1] = value[rst == 1] - self.th
		value[rst == 1] = 0

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

def compute_classification_accuracy(snn, dataset, transform):

	""" 
	Computes classification accuracy on supplied data.
	"""

	accs = []

	count = 0
	max_count = 100

	for data in dataset:

		if count == max_count:
			break

		dense_data = transform(data[0]).to_dense()
		label = data[1]

		output = snn(dense_data)

		# Max over time
		m,_= torch.max(output, 1)

		# Argmax over output units
		_, am=torch.max(m, 0)

		accs.append(label == am)

		count += 1

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
threshold = 1

# Import and transform data
dataset = tonic.datasets.hsd.SHD(save_to='./data', train=False)
transform = TonicTransform(
	min_time	= min_time,
	max_time	= max_time,
	n_samples	= n_samples,
	n_inputs	= n_inputs
)

result_dir = "./Results"

mkdir = "mkdir -p " + result_dir
sp.run(mkdir, shell=True, executable="/bin/bash")

# Import trained weights
w1 = torch.load("w1.pt", map_location=torch.device('cpu')).transpose(0, 1)
v1 = torch.load("v1.pt", map_location=torch.device('cpu')).transpose(0, 1)
w2 = torch.load("w2.pt", map_location=torch.device('cpu')).transpose(0, 1)


# # FIXED-POINT SHIFT ------------------------------------------------------------ 
# 
# filename = "fp_shift.npy"
# filename = result_dir + "/" + filename
# 
# # Fixed bit-widths
# bitwidth1 = 64
# bitwidth2 = 64
# w_bitwidth1 = 64
# w_bitwidth_fb1 = 64
# w_bitwidth2 = 64
# 
# accuracy = []
# 
# for fp_dec in range(64): 
# 
# 	# Quantize weights
# 	w1_quant = quantize.fixed_point(w1, fp_dec, w_bitwidth1)
# 	v1_quant = quantize.fixed_point(v1, fp_dec, w_bitwidth_fb1)
# 	w2_quant = quantize.fixed_point(w2, fp_dec, w_bitwidth2)
# 
# 	# Quantize threshold
# 	threshold_quant = quantize.fixed_point(threshold, fp_dec, bitwidth1)
# 
# 	# Generate the network
# 	snn = SNN(
# 		num_inputs	= n_inputs,
# 		num_hidden	= n_hidden,
# 		num_output	= n_output,
# 		threshold	= threshold_quant,
# 		alpha_shift	= alpha_shift,
# 		beta_shift	= beta_shift,
# 		w1          = w1_quant,
# 		v1          = v1_quant,
# 		w2          = w2_quant,
# 		bitwidth1	= bitwidth1,
# 		bitwidth2	= bitwidth2
# 	)
# 
# 	accuracy.append(compute_classification_accuracy(snn, dataset, transform))
# 
# with open(filename, "wb") as fp:
# 	np.save(fp, np.array(accuracy))


# WEIGHTS BIT-WIDTH FF1 --------------------------------------------------------

filename = "w1_bw.pkl"
filename = result_dir + "/" + filename

# Fixed bit-widths
bitwidth1 = 64
bitwidth2 = 64
w_bitwidth_fb1 = 64
w_bitwidth2 = 64

acc_dict = {}

for fp_dec in range(9, 13): 

	fp_key = "fp " + str(fp_dec)

	accuracy = []

	for w_bitwidth1 in range(32):

		# Quantize weights
		w1_quant = quantize.fixed_point(w1, fp_dec, w_bitwidth1)
		v1_quant = quantize.fixed_point(v1, fp_dec, w_bitwidth_fb1)
		w2_quant = quantize.fixed_point(w2, fp_dec, w_bitwidth2)

		# Quantize threshold
		threshold_quant = quantize.fixed_point(threshold, fp_dec, bitwidth1)

		# Generate the network
		snn = SNN(
			num_inputs	= n_inputs,
			num_hidden	= n_hidden,
			num_output	= n_output,
			threshold	= threshold_quant,
			alpha_shift	= alpha_shift,
			beta_shift	= beta_shift,
			w1		= w1_quant,
			v1		= v1_quant,
			w2		= w2_quant,
			bitwidth1	= bitwidth1,
			bitwidth2	= bitwidth2
		)

		accuracy.append(compute_classification_accuracy(snn, dataset, transform))

	acc_dict[fp_key] = accuracy

with open(filename, "wb") as fp:
	pickle.dump(acc_dict, fp)


# WEIGHTS BIT-WIDTH R1 ---------------------------------------------------------

filename = "v1_bw.pkl"
filename = result_dir + "/" + filename

# Fixed bit-widths
bitwidth1 = 64
bitwidth2 = 64
w_bitwidth1 = 64
w_bitwidth2 = 64

acc_dict = {}

for fp_dec in range(9, 13): 

	fp_key = "fp " + str(fp_dec)

	accuracy = []

	for w_bitwidth_fb1 in range(32):

		# Quantize weights
		w1_quant = quantize.fixed_point(w1, fp_dec, w_bitwidth1)
		v1_quant = quantize.fixed_point(v1, fp_dec, w_bitwidth_fb1)
		w2_quant = quantize.fixed_point(w2, fp_dec, w_bitwidth2)

		# Quantize threshold
		threshold_quant = quantize.fixed_point(threshold, fp_dec, bitwidth1)

		# Generate the network
		snn = SNN(
			num_inputs	= n_inputs,
			num_hidden	= n_hidden,
			num_output	= n_output,
			threshold	= threshold_quant,
			alpha_shift	= alpha_shift,
			beta_shift	= beta_shift,
			w1          = w1_quant,
			v1          = v1_quant,
			w2          = w2_quant,
			bitwidth1	= bitwidth1,
			bitwidth2	= bitwidth2
		)

		accuracy.append(compute_classification_accuracy(snn, dataset, transform))

	acc_dict[fp_key] = accuracy

with open(filename, "wb") as fp:
	pickle.dump(acc_dict, fp)



# WEIGHTS BIT-WIDTH FF2 --------------------------------------------------------

filename = "w2_bw.pkl"
filename = result_dir + "/" + filename

# Fixed bit-widths
bitwidth1 = 64
bitwidth2 = 64
w_bitwidth1 = 64
w_bitwidth_fb1 = 64

acc_dict = {}

for fp_dec in range(9, 13): 

	fp_key = "fp " + str(fp_dec)

	accuracy = []

	for w_bitwidth2 in range(32):

		# Quantize weights
		w1_quant = quantize.fixed_point(w1, fp_dec, w_bitwidth1)
		v1_quant = quantize.fixed_point(v1, fp_dec, w_bitwidth_fb1)
		w2_quant = quantize.fixed_point(w2, fp_dec, w_bitwidth2)

		# Quantize threshold
		threshold_quant = quantize.fixed_point(threshold, fp_dec, bitwidth1)

		# Generate the network
		snn = SNN(
			num_inputs	= n_inputs,
			num_hidden	= n_hidden,
			num_output	= n_output,
			threshold	= threshold_quant,
			alpha_shift	= alpha_shift,
			beta_shift	= beta_shift,
			w1          = w1_quant,
			v1          = v1_quant,
			w2          = w2_quant,
			bitwidth1	= bitwidth1,
			bitwidth2	= bitwidth2
		)

		accuracy.append(compute_classification_accuracy(snn, dataset, transform))

	acc_dict[fp_key] = accuracy

with open(filename, "wb") as fp:
	pickle.dump(acc_dict, fp)



# # NEURON BITWIDTH L1 -----------------------------------------------------------
# 
# filename = "bw1.npy"
# filename = result_dir + "/" + filename
# 
# # Fixed bit-widths
# fp_dec = 9
# bitwidth2 = 64
# w_bitwidth1 = 6
# w_bitwidth_fb1 = 5
# w_bitwidth2 = 6
# 
# # Quantize weights
# w1_quant = quantize.fixed_point(w1, fp_dec, w_bitwidth1)
# v1_quant = quantize.fixed_point(v1, fp_dec, w_bitwidth_fb1)
# w2_quant = quantize.fixed_point(w2, fp_dec, w_bitwidth2)
# 
# accuracy = []
# 
# for bitwidth1 in range(32): 
# 
# 	# Quantize threshold
# 	threshold_quant = quantize.fixed_point(threshold, fp_dec, bitwidth1)
# 
# 	# Generate the network
# 	snn = SNN(
# 		num_inputs	= n_inputs,
# 		num_hidden	= n_hidden,
# 		num_output	= n_output,
# 		threshold	= threshold_quant,
# 		alpha_shift	= alpha_shift,
# 		beta_shift	= beta_shift,
# 		w1		= w1_quant,
# 		v1		= v1_quant,
# 		w2		= w2_quant,
# 		bitwidth1	= bitwidth1,
# 		bitwidth2	= bitwidth2
# 	)
# 
# 	accuracy.append(compute_classification_accuracy(snn, dataset, transform))
# 
# with open(filename, "wb") as fp:
# 	np.save(fp, np.array(accuracy))
# 
# 
# # NEURON BITWIDTH L2 -----------------------------------------------------------
# 
# filename = "bw2.npy"
# filename = result_dir + "/" + filename
# 
# # Fixed bit-widths
# fp_dec = 9
# bitwidth1 = 64
# w_bitwidth1 = 6
# w_bitwidth_fb1 = 5
# w_bitwidth2 = 6
# 
# # Quantize weights
# w1_quant = quantize.fixed_point(w1, fp_dec, w_bitwidth1)
# v1_quant = quantize.fixed_point(v1, fp_dec, w_bitwidth_fb1)
# w2_quant = quantize.fixed_point(w2, fp_dec, w_bitwidth2)
# 
# accuracy = []
# 
# for bitwidth2 in range(32): 
# 
# 	# Quantize threshold
# 	threshold_quant = quantize.fixed_point(threshold, fp_dec, bitwidth1)
# 
# 	# Generate the network
# 	snn = SNN(
# 		num_inputs	= n_inputs,
# 		num_hidden	= n_hidden,
# 		num_output	= n_output,
# 		threshold	= threshold_quant,
# 		alpha_shift	= alpha_shift,
# 		beta_shift	= beta_shift,
# 		w1		= w1_quant,
# 		v1		= v1_quant,
# 		w2		= w2_quant,
# 		bitwidth1	= bitwidth1,
# 		bitwidth2	= bitwidth2
# 	)
# 
# 	accuracy.append(compute_classification_accuracy(snn, dataset, transform))
# 
# with open(filename, "wb") as fp:
# 	np.save(fp, np.array(accuracy))

# # Exponential shifts
# alpha_shift_0 = alpha_shift
# beta_shift_0 = beta_shift
# 
# # Fixed bit-widths
# fp_dec = 9
# bitwidth1 = 10
# bitwidth2 = 8
# w_bitwidth1 = 6
# w_bitwidth_fb1 = 5
# w_bitwidth2 = 6
# 
# # Quantize weights
# w1_quant = quantize.fixed_point(w1, fp_dec, w_bitwidth1)
# v1_quant = quantize.fixed_point(v1, fp_dec, w_bitwidth_fb1)
# w2_quant = quantize.fixed_point(w2, fp_dec, w_bitwidth2)
# 
# # Quantize threshold
# threshold_quant = quantize.fixed_point(threshold, fp_dec, bitwidth1)
# 
# # # ALPHA1 SHIFT -----------------------------------------------------------------
# # 
# # filename = "alpha1.npy"
# # filename = result_dir + "/" + filename
# # 
# # accuracy = []
# # 
# # for alpha_shift in range(10): 
# # 
# # 	# Generate the network
# # 	snn = SNN(
# # 		num_inputs	= n_inputs,
# # 		num_hidden	= n_hidden,
# # 		num_output	= n_output,
# # 		threshold	= threshold_quant,
# # 		alpha_shift	= alpha_shift,
# # 		beta_shift	= beta_shift_0,
# # 		alpha_shift2	= alpha_shift_0,
# # 		beta_shift2	= beta_shift_0,
# # 		w1		= w1_quant,
# # 		v1		= v1_quant,
# # 		w2		= w2_quant,
# # 		bitwidth1	= bitwidth1,
# # 		bitwidth2	= bitwidth2
# # 	)
# # 
# # 	accuracy.append(compute_classification_accuracy(snn, dataset, transform))
# # 
# # with open(filename, "wb") as fp:
# # 	np.save(fp, np.array(accuracy))
# 
# 
# # ALPHA2 SHIFT -----------------------------------------------------------------
# filename = "alpha2.npy"
# filename = result_dir + "/" + filename
# 
# accuracy = []
# 
# for alpha_shift in range(32): 
# 
# 	# Generate the network
# 	snn = SNN(
# 		num_inputs	= n_inputs,
# 		num_hidden	= n_hidden,
# 		num_output	= n_output,
# 		threshold	= threshold_quant,
# 		alpha_shift	= alpha_shift_0,
# 		beta_shift	= beta_shift_0,
# 		alpha_shift2	= alpha_shift,
# 		beta_shift2	= beta_shift_0,
# 		w1		= w1_quant,
# 		v1		= v1_quant,
# 		w2		= w2_quant,
# 		bitwidth1	= bitwidth1,
# 		bitwidth2	= bitwidth2
# 	)
# 
# 	accuracy.append(compute_classification_accuracy(snn, dataset, transform))
# 
# with open(filename, "wb") as fp:
# 	np.save(fp, np.array(accuracy))
# 
# 
# # # BETA1 SHIFT -----------------------------------------------------------------
# # filename = "beta1.npy"
# # filename = result_dir + "/" + filename
# # 
# # accuracy = []
# # 
# # for beta_shift in range(10): 
# # 
# # 	# Generate the network
# # 	snn = SNN(
# # 		num_inputs	= n_inputs,
# # 		num_hidden	= n_hidden,
# # 		num_output	= n_output,
# # 		threshold	= threshold_quant,
# # 		alpha_shift	= alpha_shift_0,
# # 		beta_shift	= beta_shift,
# # 		alpha_shift2	= alpha_shift_0,
# # 		beta_shift2	= beta_shift_0,
# # 		w1		= w1_quant,
# # 		v1		= v1_quant,
# # 		w2		= w2_quant,
# # 		bitwidth1	= bitwidth1,
# # 		bitwidth2	= bitwidth2
# # 	)
# # 
# # 	accuracy.append(compute_classification_accuracy(snn, dataset, transform))
# # 
# # with open(filename, "wb") as fp:
# # 	np.save(fp, np.array(accuracy))
# 
# # BETA2 SHIFT -----------------------------------------------------------------
# filename = "beta2.npy"
# filename = result_dir + "/" + filename
# 
# accuracy = []
# 
# for beta_shift in range(32): 
# 
# 	# Generate the network
# 	snn = SNN(
# 		num_inputs	= n_inputs,
# 		num_hidden	= n_hidden,
# 		num_output	= n_output,
# 		threshold	= threshold_quant,
# 		alpha_shift	= alpha_shift_0,
# 		beta_shift	= beta_shift_0,
# 		alpha_shift2	= alpha_shift_0,
# 		beta_shift2	= beta_shift,
# 		w1		= w1_quant,
# 		v1		= v1_quant,
# 		w2		= w2_quant,
# 		bitwidth1	= bitwidth1,
# 		bitwidth2	= bitwidth2
# 	)
# 
# 	accuracy.append(compute_classification_accuracy(snn, dataset, transform))
# 
# with open(filename, "wb") as fp:
# 	np.save(fp, np.array(accuracy))
