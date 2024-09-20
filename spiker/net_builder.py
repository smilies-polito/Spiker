import re
import json
import logging
import torch
import torch.nn as nn
import snntorch as snn


class SNN(nn.Module):

	def __init__(self, net_dict):

		super(SNN, self).__init__()

		self.n_cycles = net_dict["n_cycles"]

		self.layers = nn.ModuleDict()

		self.syn = {}
		self.mem = {}
		self.spk = {}

		self.syn_rec = {}
		self.mem_rec = {}
		self.spk_rec = {}

		self.build_snn(net_dict)

	
	def build_snn(self, net_dict):

		first = True

		for key in net_dict:
 
			if "layer" in key:

				idx = str(self.extract_index(key) + 1)

				if first:
					self.layers["fc" + idx] = nn.Linear(
						in_features		= net_dict["n_inputs"],
						out_features	= net_dict[key]["n_neurons"],
						bias 			= False
					)

					n_inputs_next = net_dict[key]["n_neurons"]

					first = False

				else:

					self.layers["fc" + idx] = nn.Linear(
						in_features		= n_inputs_next,
						out_features	= net_dict[key]["n_neurons"],
						bias 			= False
					)

					n_inputs_next = net_dict[key]["n_neurons"]

				if net_dict[key]["neuron_model"] == "if":

					name = "if" + idx
					
					self.layers[name] = snn.Leaky(
						beta			= 0.,
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"],
						reset_mechanism = net_dict[key]["reset_mechanism"]
					)

				elif net_dict[key]["neuron_model"] == "lif":

					name = "lif" + idx

					self.layers[name] = snn.Leaky(
						beta			= net_dict[key]["beta"],
						learn_beta		= net_dict[key]["learn_beta"],
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"],
						reset_mechanism = net_dict[key]["reset_mechanism"]
					)

				elif net_dict[key]["neuron_model"] == "syn":

					name = "syn" + idx

					self.layers[name] = snn.Synaptic(
						alpha			= net_dict[key]["alpha"],
						learn_alpha		= net_dict[key]["learn_alpha"],
						beta			= net_dict[key]["beta"],
						learn_beta		= net_dict[key]["learn_beta"],
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"],
						reset_mechanism = net_dict[key]["reset_mechanism"]
					)

				elif net_dict[key]["neuron_model"] == "rif":

					name = "rif" + idx

					self.layers[name] = snn.RLeaky(
						linear_features	= net_dict[key]["n_neurons"],
						beta			= 0.,
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"],
						reset_mechanism = net_dict[key]["reset_mechanism"]
					)

				elif net_dict[key]["neuron_model"] == "rlif":

					name = "rlif" + idx

					self.layers[name] = snn.RLeaky(
						linear_features	= net_dict[key]["n_neurons"],
						beta			= net_dict[key]["beta"],
						learn_beta		= net_dict[key]["learn_beta"],
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"],
						reset_mechanism = net_dict[key]["reset_mechanism"]
					)

				elif net_dict[key]["neuron_model"] == "rsyn":

					name = "rsyn" + idx

					self.layers[name] = snn.RSynaptic(
						linear_features	= net_dict[key]["n_neurons"],
						alpha			= net_dict[key]["alpha"],
						learn_alpha		= net_dict[key]["learn_alpha"],
						beta			= net_dict[key]["beta"],
						learn_beta		= net_dict[key]["learn_beta"],
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"],
						reset_mechanism = net_dict[key]["reset_mechanism"]
					)

				else:
					raise ValueError(
						"Invalid neuron model. "\
						"Pick one between " \
						"if, lif, syn, rif, rlif, rsyn."
					)

	def reset(self):

		for layer in self.layers:

			idx = str(self.extract_index(layer))

			if "fc" not in layer:

				self.mem_rec[layer] = []
				self.syn_rec[layer] = []
				self.spk_rec[layer] = []

				if layer == "if" + idx:
					self.mem[layer] = self.layers[layer].reset_mem()

				elif layer == "lif" + idx:
					self.mem[layer] = self.layers[layer].reset_mem()

				elif layer == "syn" + idx:
					self.syn[layer], self.mem[layer] = self.layers[layer].\
														reset_mem()

				elif layer == "rif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer].\
													reset_mem()

				elif layer == "rlif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer].\
													reset_mem()

				elif layer == "rsyn" + idx:
					self.spk[layer], self.syn[layer], self.mem[layer] = \
							self.layers[layer].reset_mem()

	def record(self, layer):

		if not "fc" in layer:
			self.mem_rec[layer].append(self.mem[layer])
			self.spk_rec[layer].append(self.spk[layer])

			if "syn" in layer:
				self.syn_rec[layer].append(self.syn[layer])

	def stack_rec(self):

			for layer in self.layers:

				if not "fc" in layer:
					self.mem_rec[layer] = torch.stack(self.mem_rec[layer], 
											dim=0)
					self.spk_rec[layer] = torch.stack(self.spk_rec[layer], 
											dim=0)

					if "syn" in layer:
						self.syn_rec[layer] = torch.stack(self.syn_rec[layer], 
											dim=0)

				
	def extract_index(self, layer_name):

		index = re.findall(r'\d+', layer_name)

		if len(index) != 1:

			error_msg = "Invalid layer name: " + str(layer_name)
			error_msg += ". Use \"layer_\" + <integer layer index>\n"

			raise ValueError(error_msg)
		else:
			return int(index[0])



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
				self.record(layer)

		self.stack_rec()

class NetBuilder:

	def __init__(self, net_dict):
		

		self.default_dict = {

			"n_cycles"				: 73,
			"n_inputs"				: 40,

			"layer_0"	: {
				
				"neuron_model"		: "lif",
				"n_neurons"			: 128,
				"alpha"				: 0.9,
				"learn_alpha"		: False,
				"beta"				: 0.9375,
				"learn_beta"		: False,
				"threshold"			: 1.,
				"learn_threshold"	: False,
				"reset_mechanism"	: "subtract"
			},

			"layer_1"	: {
				
				"neuron_model"		: "lif",
				"n_neurons"			: 10,
				"alpha"				: 0.9,
				"learn_alpha"		: False,
				"beta"				: 0.9375,
				"learn_beta"		: False,
				"threshold"			: 1.,
				"learn_threshold"	: False,
				"reset_mechanism"	: "none"
			}
		}

		self.allowed_keys = self.select_keys()
		self.supported_models = ["if", "lif", "syn", "rif", "rlif", "rsyn"]

		self.has_alpha = {
			"if"	: False,
			"lif"	: False,
			"syn"	: True,
			"rif"	: False,
			"rlif"	: False,
			"rsyn"	: True,
		}

		self.has_beta = {
			"if"	: False,
			"lif"	: True,
			"syn"	: True,
			"rif"	: False,
			"rlif"	: True,
			"rsyn"	: True,
		}

		self.supported_resets = ["zero", "subtract", "none"]

		self.net_dict = self.parse_config(net_dict)

	def build(self):

		snn = SNN(self.net_dict)

		log_message = "Network ready: " + str(snn) + "\n"
		logging.info(log_message)

		return snn


	def select_keys(self):

		keywords = self.default_dict.keys()

		allowed_keys = []

		for k in keywords:

			k = re.sub(r'\d+', '', k)

			if k not in allowed_keys:
				allowed_keys.append(k)

		return allowed_keys



	def parse_config(self, net_dict):

		parsed_dict = {}

		for key in net_dict:

			if any([allowed in key for allowed in self.allowed_keys]):

				if "layer" not in key:

					if type(net_dict[key]) is not int:
						raise ValueError("Error, " + key + " must be an "
							"integer value\n")

					parsed_dict[key] = net_dict[key]

				else:

					parsed_dict[key] = {}
					layer = net_dict[key]

					if "n_neurons" in layer.keys():

						if not isinstance(layer["n_neurons"], int):
							raise ValueError("Number of neurons must be "
									"integer.\n")

						else:
							parsed_dict[key]["n_neurons"] = \
									layer["n_neurons"]

					else:
						parsed_dict[key]["n_neurons"] = \
							self.default_dict["layer_0"]["n_neurons"]


					if "neuron_model" in layer.keys():

						if layer["neuron_model"] not in self.supported_models:
							raise ValueError("Unsupported neuron model. "
									"Choose one between " +
									str(self.supported_models) + "\n")

						else:
							parsed_dict[key]["neuron_model"] = \
									layer["neuron_model"]

					else:
						parsed_dict[key]["neuron_model"] = \
							self.default_dict["layer_0"]["neuron_model"]

					if "threshold" in layer.keys():

						if not isinstance(layer["threshold"], (int, float)):
							raise ValueError("Treshold must be numeric.\n")

						else:
							parsed_dict[key]["threshold"] = layer["threshold"]

					else:
						parsed_dict[key]["threshold"] = \
							self.default_dict["layer_0"]["threshold"]

					if "learn_threshold" in layer.keys():

						if not isinstance(layer["learn_threshold"], bool):
							raise ValueError("learn_threshold must be "
									"boolean.\n")

						else:
							parsed_dict[key]["learn_threshold"] = \
									layer["learn_threshold"]

					else:
						parsed_dict[key]["learn_threshold"] = \
							self.default_dict["layer_0"]["learn_threshold"]

					if "reset_mechanism" in layer.keys():

						if layer["reset_mechanism"] not in \
						self.supported_resets:
							raise ValueError("Invalid reset mechanism. "
									"Choose one between " +
									str(self.supported_resets) + "\n")

						else:
							parsed_dict[key]["reset_mechanism"] = \
									layer["reset_mechanism"]

					else:
						parsed_dict[key]["reset_mechanism"] = \
							self.default_dict["layer_0"]["reset_mechanism"]


					if self.has_alpha[parsed_dict[key]["neuron_model"]]:

						if "alpha" in layer.keys():

							if not isinstance(layer["alpha"], float):
								raise ValueError("Alpha decay must be "
										"float\n")

							elif (layer["alpha"] < 0. or layer["alpha"] > 1.):
								raise ValueError("Alpha decay must be "
										"between 0 and 1\n")

							else:
								parsed_dict[key]["alpha"] = \
										layer["alpha"]

						else:
							parsed_dict[key]["alpha"] = \
								self.default_dict["layer_0"]["alpha"]

						if "learn_alpha" in layer.keys():

							if not isinstance(layer["learn_alpha"], bool):
								raise ValueError("learn_alpha must be "
										"boolean.\n")

							else:
								parsed_dict[key]["learn_alpha"] = \
										layer["learn_alpha"]

						else:
							parsed_dict[key]["learn_alpha"] = \
								self.default_dict["layer_0"]["learn_alpha"]

					if self.has_beta[parsed_dict[key]["neuron_model"]]:

						if "beta" in layer.keys():

							if not isinstance(layer["beta"], float):
								raise ValueError("Alpha decay must be "
										"float\n")

							elif (layer["beta"] < 0. or layer["beta"] > 1.):
								raise ValueError("Alpha decay must be "
										"between 0 and 1\n")

							else:
								parsed_dict[key]["beta"] = \
										layer["beta"]

						else:
							parsed_dict[key]["beta"] = \
								self.default_dict["layer_0"]["beta"]

						if "learn_beta" in layer.keys():

							if not isinstance(layer["learn_beta"], bool):
								raise ValueError("learn_beta must be "
										"boolean.\n")

							else:
								parsed_dict[key]["learn_beta"] = \
										layer["learn_beta"]

						else:
							parsed_dict[key]["learn_beta"] = \
								self.default_dict["layer_0"]["learn_beta"]

						
		if "n_cycles" not in parsed_dict:
			parsed_dict["n_cycles"] = self.default_dict["n_cycles"]

		if "n_inputs" not in parsed_dict:
			parsed_dict["n_inputs"] = self.default_dict["n_inputs"]

		at_least_one_layer = False
		for key in parsed_dict:
			if "layer_" in key:
				at_least_one_layer = True

		if not at_least_one_layer:

			for key in self.default_dict:

				if "layer_" in key:

					parsed_dict[key] = self.default_dict[key]

		log_message = "Network configured: \n"
		log_message += json.dumps(parsed_dict, indent = 4) + "\n"

		logging.info(log_message)

		return parsed_dict




if __name__ == "__main__": 

	from net_dict import net_dict

	logging.basicConfig(level=logging.INFO)

	net_builder = NetBuilder(net_dict)

	snn = net_builder.build()	
