from math import log2
import torch
import numpy as np

from .vhdl.layer import Layer
from .vhdl.network import Network, FullAccelerator

class VhdlGenerator:

	def __init__(self, net, optim_config):

		self.net = net
		self.optim_config = optim_config

		# When learning is enabled, the SNN weights are already in
		# fixed-point integer form (the STSFTrainer pre-quantized them).
		# Validate that the optim_config agrees with the bitwidths the
		# learning block was configured with (single source of truth).
		if self.net.learning is not None:
			learn_bw     = self.net.learning["bw"]
			learn_fp_dec = self.net.learning["fp_dec"]
			if optim_config["fp_dec"] != learn_fp_dec \
					or optim_config["neurons_bw"] != learn_bw \
					or optim_config["weights_bw"] != learn_bw:
				raise ValueError(
					"learning mode requires bitwidth_config to match the "
					f"learning block (fp_dec={learn_fp_dec}, "
					f"neurons_bw=weights_bw={learn_bw}); "
					f"got {optim_config}")
			# Pull the trainer-published constants off the SNN.
			if not hasattr(self.net, "sel_idx_array") \
					or not hasattr(self.net, "neuron_constants"):
				raise ValueError(
					"learning mode requires STSFTrainer.train() to have "
					"run first (so snn.sel_idx_array and "
					"snn.neuron_constants are populated)")
			# update_every_n is carried on a cycles_cnt_bitwidth-wide
			# port (the testbench convention drives it with N-1).
			from math import log2 as _log2
			from .vhdl.utils import ceil_pow2 as _ceil_pow2
			cycles_cnt_bw = int(_log2(_ceil_pow2(
				self.net.n_cycles + 1))) + 1
			max_n = 2 ** cycles_cnt_bw
			update_every = self.net.learning["update_every"]
			if not 1 <= update_every <= max_n:
				raise ValueError(
					f"learning update_every={update_every} does "
					f"not fit the {cycles_cnt_bw}-bit "
					"update_every_n port (valid range: 1.."
					f"{max_n} for n_cycles="
					f"{self.net.n_cycles})")

		self.input_size = self.input_size(list(self.net.layers)[0])
		self.output_size = self.output_size(list(self.net.layers)[-2])

	def generate(self, functional = True, interface = False, debug = False):

		learning_block = self.net.learning
		n_classes = self.output_size if learning_block is not None else None

		# On-chip learning supports exactly one configuration -- the one
		# the hand-coded Spiker-LL accelerators use (functional design,
		# inferred-BRAM memories, no wrapper, no debug taps). Reject the
		# other modes explicitly instead of emitting untested VHDL.
		if learning_block is not None and (
				interface or debug or not functional):
			raise ValueError(
				"on-chip learning supports only the default "
				"generate() configuration (functional=True, "
				"interface=False, debug=False); got "
				f"functional={functional}, interface={interface}, "
				f"debug={debug}")

		# Set (or reset) the package-level learning switch before any
		# VHDL object is constructed -- see SpikerPackage.learning_mode.
		from .vhdl.spiker_pkg import SpikerPackage
		SpikerPackage.learning_mode = learning_block is not None

		vhdl_net = Network(
			self.net.n_cycles,
			learning=learning_block,
			n_classes=n_classes,
			debug=debug,
		)
		self.functional = functional

		# Track which trainable index we're currently on so init_layer can
		# tell hidden (idx 0) from output (idx 1).
		self._trainable_idx = 0
		self._n_output_neurons = self.output_size

		for layer in self.net.layers:

			if "fc" in layer:

				ff_w = self.extract_weights(layer)

			else:

				vhdl_net.add(self.init_layer(layer, ff_w))

		# Resolve cross-layer wiring (pred_spikes feedback, voter
		# instantiation). No-op when learning is disabled.
		vhdl_net.finalize()

		if not interface:

			return vhdl_net

		else:

			return FullAccelerator(vhdl_net, self.input_size,
					self.output_size)


	def input_size(self, layer):

			if "fc" in layer:

				ff_w = self.extract_weights(layer)

				return ff_w.shape[1]

			raise ValueError("Cannot compute size. I need a linear layer")

	def output_size(self, layer):

			if "fc" in layer:

				ff_w = self.extract_weights(layer)

				return ff_w.shape[0]

			raise ValueError("Cannot compute size. I need a linear layer")

	def init_layer(self, layer, ff_w):

		th = np.repeat(self.extract_threshold(layer), ff_w.shape[0])
		beta_shift = self.extract_beta(layer)
		reset = self.extract_reset(layer)
		fb_w = self.extract_weights(layer)

		if not fb_w:
			fb_w = torch.zeros((ff_w.shape[0], ff_w.shape[0])).numpy()

		learning = self.net.learning
		# In learning mode the SNN's weights and thresholds are already
		# fixed-point integers (STSFTrainer pre-quantizes them), so the
		# Layer-level fp_decimals scaling must be skipped.
		fp_decimals = 0 if learning is not None \
			else self.optim_config["fp_dec"]

		trainable_kwargs = {}
		if learning is not None:
			# Map the two layers to (hidden, output) roles in the order
			# they appear in the SNN's ModuleDict.
			role = "hidden" if self._trainable_idx == 0 else "output"
			trainable_kwargs = {
				"trainable":           True,
				"role":                role,
				"output_lr":           learning["lr"],
				"output_loss_value":   learning["loss_value"],
				"output_const_value":  learning.get("output_const_value"),
				"hw_fp_dec":           learning["fp_dec"],
			}
			if role == "hidden":
				trainable_kwargs["sel_idx"] = self.net.sel_idx_array
				trainable_kwargs["neuron_constants"] = \
					self.net.neuron_constants
				trainable_kwargs["n_output_neurons"] = self._n_output_neurons
			self._trainable_idx += 1

		return Layer(
			label		= layer,
			w_exc		= ff_w,
			w_inh		= fb_w,
			v_th		= th,
			bitwidth	= self.optim_config["neurons_bw"],
			fp_decimals	= fp_decimals,
			w_inh_bw	= self.optim_config["weights_bw"],
			w_exc_bw	= self.optim_config["weights_bw"],
			shift		= beta_shift,
			reset		= reset,
			functional	= self.functional,
			**trainable_kwargs,
		)


	def extract_weights(self, layer):

		if "weight" in dir(self.net.layers[layer]):

			return self.net.layers[layer].weight.data.cpu().numpy()

		elif "recurrent" in dir(self.net.layers[layer]):

			return self.net.layers[layer].recurrent.weight.data.cpu().numpy()


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

			elif reset == "none":
				return "none"

			else:
				raise ValueError("Reset type not supported")


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
