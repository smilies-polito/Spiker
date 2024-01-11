import snntorch as snn
import torch

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

	def init_neurons(self):
		self.mem = torch.zeros(self.n_neurons)
		self.syn = torch.zeros(self.n_neurons)
		self.out = torch.zeros(self.n_neurons)

	def exp_decay(self, value, decay_rate):
		return value*decay_rate

	def reset(self, value):

		# Subtractive reset
		rst = self.out.detach()
		value[rst == 1] = value[rst == 1] - self.th

	def spike_fn(self, mthr):
		self.out = torch.zeros_like(mthr)
		self.out[mthr > 0] = 1.0


	def __call__(self, input_current):

		mthr = self.mem - self.th

		self.spike_fn(mthr)

		self.syn = self.exp_decay(self.syn, self.alpha) + input_current
		self.mem = self.exp_decay(self.mem, self.beta) + self.syn

		if not self.readout:
			self.reset(self.mem)


		return self.out, self.syn, self.mem

alpha 		= 1 - 2**-4
beta 		= 1 - 2**-3
threshold	= 1


my_lif = LIF(
	alpha		= alpha,
	beta		= beta,
	threshold	= threshold,
	n_neurons	= 1,
	readout		= False
)

snntorch_lif = snn.Synaptic(
	alpha		= alpha,
	beta		= beta,
	threshold	= threshold
)

spikes = [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1.]

syn, snntorch_mem = snntorch_lif.init_synaptic()

for s in spikes:
	_, _, my_mem = my_lif(s)
	_, syn, snntorch_mem = snntorch_lif(torch.tensor(s), syn, snntorch_mem)
	out = str(my_mem) + "\t" + str(snntorch_mem)
	print(out)
