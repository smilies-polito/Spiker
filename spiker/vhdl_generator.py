from math import log2
import torch
import numpy as np

from .vhdl.network import Network, Layer

class VhdlGenerator:

	def __init__(self, net, optim_config):
		
		self.net = net
		self.optim_config = optim_config

	def generate(self):

		vhdl_net = Network(self.net.n_cycles)

		for layer in self.net.layers:

			if "fc" in layer:

				ff_w = self.extract_weights(layer)

			else:

				vhdl_net.add(self.init_layer(layer, ff_w))

		return vhdl_net
		

	def init_layer(self, layer, ff_w):

		th = np.repeat(self.extract_threshold(layer), ff_w.shape[0]) 
		beta_shift = self.extract_beta(layer)
		reset = self.extract_reset(layer)
		fb_w = self.extract_weights(layer)

		if not fb_w:
			fb_w = torch.zeros((ff_w.shape[0], ff_w.shape[0])).numpy()

		return Layer(
			label		= layer,
			w_exc		= ff_w,
			w_inh		= fb_w,
			v_th		= th,
			bitwidth	= self.optim_config["neurons_bw"],
			fp_decimals	= self.optim_config["fp_dec"],
			w_inh_bw	= self.optim_config["weights_bw"],
			w_exc_bw	= self.optim_config["weights_bw"],
			shift		= beta_shift,
			reset		= "subtractive",
			functional	= True
		)


	def extract_weights(self, layer):

		if "weight" in dir(self.net.layers[layer]):

			return self.net.layers[layer].weight.data.numpy()

		elif "recurrent" in dir(self.net.layers[layer]):

			return self.net.layers[layer].recurrent.weight.data.numpy()


	def extract_threshold(self, layer):

		if "threshold" in dir(self.net.layers[layer]):

			return np.array([self.net.layers[layer].threshold.data.item()])

	def extract_reset(self, layer):

		if "reset_mechanism" in dir(self.net.layers[layer]):
			
			reset = self.net.layers[layer].reset_mechanism

			if reset == "subtract":
				return "subtractive"

			elif reset == "zero":
				return "fixed"

			else:
				return "fixed"


	def extract_alpha(self, layer):

		if "alpha" in dir(self.net.layers[layer]):

			alpha = self.net.layers[layer].alpha.data.item()

			return self.pow2_shift(1 - alpha)


	def extract_beta(self, layer):

		if "beta" in dir(self.net.layers[layer]):

			beta = self.net.layers[layer].beta.data.item()

			return self.pow2_shift(1 - beta)


	def pow2_shift(self, value):
		return int(abs(log2(value)))
