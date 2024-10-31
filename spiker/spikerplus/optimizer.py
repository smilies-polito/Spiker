import logging
import json
import torch
from tabulate import tabulate
import numpy as np

from .net_builder import SNN
from .trainer import Trainer

class Quantizer:

	def fixed_point(self, value, fp_dec, bitwidth):

		quant = value * 2**fp_dec

		return self.saturated_int(quant, bitwidth)

	def saturated_int(self, value, bitwidth):
		return self.saturate(self.to_int(value), bitwidth)

	def saturate(self, value, bitwidth):

		if type(value).__module__ == np.__name__ or \
		type(value).__module__ == torch.__name__:

			value[value > 2**(bitwidth-1)-1] = \
				2**(bitwidth-1)-1
			value[value < -2**(bitwidth-1)] = \
				-2**(bitwidth-1)

			return value.float()

		else:

			if value > 2**(bitwidth-1)-1:
				value = 2**(bitwidth-1)-1

			elif value < -2**(bitwidth-1):
				value = -2**(bitwidth-1)

			return float(value)

	def to_int(self, value):

		if type(value).__module__ == np.__name__:
			quant = value.astype(int).astype(float)

		elif type(value).__module__ == torch.__name__:
			quant = value.type(torch.int64).float()

		else:
			quant = float(int(value))

		return quant



class QuantSNN(SNN):

	def __init__(self, net_dict, neurons_bw):

		super().__init__(net_dict)

		self.neurons_bw = neurons_bw

		self.quantizer = Quantizer()


	def forward(self, input_spikes):

		self.reset()

		cur = {}

		if input_spikes.shape[0] != self.n_cycles:
			logging.warning("Input data have a time dimension different from "\
					"the network's number of steps. It's ok at this level, "\
					"but remember to use a suitable number of steps in the "\
					"vhdl generator")

		for step in range(input_spikes.shape[0]):

			first = True

			for layer in self.layers:

				idx = str(self.extract_index(layer))
				
				if "fc" in layer:

					if first:
						cur[layer] = self.layers[layer](input_spikes[step])
						first = False

					else:
						cur[layer] = self.layers[layer](self.spk[prev_layer])

				elif layer == "if" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[prev_layer], self.mem[layer])

				elif layer == "lif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[prev_layer], self.mem[layer])

				elif layer == "syn" + idx:
					self.spk[layer], self.syn[layer], self.mem[layer] = \
							self.layers[layer](cur[prev_layer], self.syn[layer],
							self.mem[layer])

				elif layer == "rif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[prev_layer], self.spk[layer], self.mem[layer])

				elif layer == "rlif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[prev_layer], self.spk[layer], self.mem[layer])

				elif layer == "rsyn" + idx:
					self.spk[layer], self.syn[layer], self.mem[layer] = \
							self.layers[layer](cur[prev_layer], self.spk[layer], 
							self.syn[layer], self.mem[layer])

				prev_layer = layer

				self.quantize(layer)

				self.record(layer)

		self.stack_rec()

	def quantize(self, layer):

		if not "fc" in layer:
			self.mem[layer] = self.quantizer.saturated_int(
					self.mem[layer], self.neurons_bw)

			if "syn" in layer:
				self.syn[layer] = self.quantizer.saturated_int(
					self.syn[layer], self.neurons_bw)



class Optimizer(Trainer):

	def __init__(self, net, net_dict, optim_config, readout_type = "mem"):

		super().__init__(net, readout_type)

		self.default_config = {

			"weights_bw"	: {
				"min"	: 4,
				"max"	: 8
			},

			"neurons_bw"	: {
				"min"	: 4,
				"max"	: 10
			},

			"fp_dec"	: {
				"min"	: 2,
				"max"	: 3
			}
		}

		self.allowed_keys = self.default_config.keys()

		self.quantizer = Quantizer()

		self.state_dict = net.state_dict()
		self.net_dict = net_dict

		self.optim_config = self.parse_config(optim_config)

		if torch.cuda.is_available():
			self.device = torch.device("cuda")

		else:
			self.device = torch.device("cpu")
		

	def parse_config(self, optim_config):

		optim_dict = {}

		for key in optim_config:

			if key in self.allowed_keys:

				if "min" in optim_config[key]:

					if not isinstance(optim_config[key]["min"], int):
						raise ValueError("Range specifiers must be integers\n")

					else:
						min_value = optim_config[key]["min"]

				else:
					min_value = self.default_config[key]["min"]

				if "max" in optim_config[key]:

					if not isinstance(optim_config[key]["max"], int):
						raise ValueError("Range specifiers must be integers\n")

					else:
						max_value = optim_config[key]["max"]

				else:
					max_value = self.default_config[key]["max"]


				optim_dict[key] = [i for i in range(min_value, max_value + 1)]


		log_message = "Optimizer configured: \n"
		log_message += json.dumps(optim_dict, indent = 4)
		logging.info(log_message)

		return optim_dict

	def optimize(self, dataloader):


		headers = ["Fixed-point decimals", "Neurons' bitwidth", 
					"Weights bitwidth", "Loss", "Accuracy"]
		table = []

		for fp_dec in self.optim_config["fp_dec"]:

			for w_bw in self.optim_config["neurons_bw"]:

				for neuron_bw in self.optim_config["weights_bw"]:

					self.build_quant_snn(w_bw, neuron_bw, fp_dec)

					loss, acc = self.evaluate(dataloader)

					log_message = "\nLoss: " + "{:.2f}".format(loss) + "\n"
					log_message += "Acc: " + "{:.2f}".format(acc*100) + "%\n"
					logging.info(log_message)

					table.append([
						str(fp_dec),
						str(neuron_bw),
						str(w_bw),
						str(loss),
						"{:.2f}".format(acc*100) + "%"
					])

		table = "\n" + tabulate(table, headers = headers, tablefmt = "grid")

		logging.info(table)


	def build_quant_snn(self, weights_bw, neurons_bw, fp_dec):

		self.net = QuantSNN(self.net_dict, neurons_bw)

		quant_state_dict = self.state_dict.copy()

		for key in quant_state_dict.keys():

			if "weight" in key:

				quant_state_dict[key] = self.quantizer.fixed_point(
						quant_state_dict[key], fp_dec, weights_bw)

			elif "threshold" in key:

				quant_state_dict[key] = self.quantizer.fixed_point(
						quant_state_dict[key], fp_dec, neurons_bw)

			self.net.load_state_dict(quant_state_dict)

		self.net.to(self.device)

		log_message = "Network ready:\n"
		log_message += "Fixed-point decimals: " + str(fp_dec) + "\n"
		log_message += "Neurons bitwidth: " + str(neurons_bw) + "\n"
		log_message += "Weights bitwidth: " + str(weights_bw) + "\n"
		logging.info(log_message)
