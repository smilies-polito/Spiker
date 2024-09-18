import re
import torch
import torch.nn as nn
import snntorch as snn


class SNN(nn.Module):

	def __init__(self, net_dict):

		super(SNN, self).__init__()

		self.layers = nn.ModuleDict()

		self.syn = {}
		self.mem = {}
		self.spk = {}

		self.syn_rec = {}
		self.mem_rec = {}
		self.spk_rec = {}

		self.build_snn(net_dict)

	
	def build_snn(self, net_dict):

		for key in net_dict:
 
			if "layer" in key:

				idx = str(self.extract_index(key) + 1)

				self.layers["fc" + idx] = nn.Linear(
					in_features		= net_dict["n_inputs"],
					out_features	= net_dict[key]["n_neurons"],
					bias 			= False
				)

				if net_dict[key]["neuron_model"] == "if":

					name = "if" + idx
					
					self.layers[name] = snn.Leaky(
						beta			= 0.,
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"]
					)

				elif net_dict[key]["neuron_model"] == "lif":

					name = "lif" + idx

					self.layers[name] = snn.Leaky(
						beta			= net_dict[key]["beta"],
						learn_beta		= net_dict[key]["learn_beta"],
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"]
					)

				elif net_dict[key]["neuron_model"] == "syn":

					name = "syn" + idx

					self.layers[name] = snn.Synaptic(
						alpha			= net_dict[key]["alpha"],
						learn_alpha		= net_dict[key]["learn_alpha"],
						beta			= net_dict[key]["beta"],
						learn_beta		= net_dict[key]["learn_beta"],
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"]
					)

				elif net_dict[key]["neuron_model"] == "rif":

					name = "rif" + idx

					self.layers[name] = snn.RLeaky(
						linear_features	= net_dict[key]["n_neurons"],
						beta			= 0.,
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"]
					)

				elif net_dict[key]["neuron_model"] == "rlif":

					name = "rlif" + idx

					self.layers[name] = snn.RLeaky(
						linear_features	= net_dict[key]["n_neurons"],
						beta			= net_dict[key]["beta"],
						learn_beta		= net_dict[key]["learn_beta"],
						threshold		= net_dict[key]["threshold"],
						learn_threshold	= net_dict[key]["learn_threshold"]
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
						learn_threshold	= net_dict[key]["learn_threshold"]
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

		for step in range(input_spikes.shape[0]):

			for layer in self.layers:

				idx = str(self.extract_index(layer))
				
				if "fc" in layer:

					cur[layer] = self.layers[layer](input_spikes[step])
					fc_layer = layer

				elif layer == "if" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[fc_layer], self.mem[layer])

				elif layer == "lif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[fc_layer], self.mem[layer])

				elif layer == "syn" + idx:
					self.spk[layer], self.syn[layer], self.mem[layer] = \
							self.layers[layer](cur[fc_layer], self.syn[layer],
							self.mem[layer])

				elif layer == "rif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[fc_layer], self.spk[layer], self.mem[layer])

				elif layer == "rlif" + idx:
					self.spk[layer], self.mem[layer] = self.layers[layer]\
							(cur[fc_layer], self.spk[layer], self.mem[layer])

				elif layer == "rsyn" + idx:
					self.spk[layer], self.syn[layer], self.mem[layer] = \
							self.layers[layer](cur[fc_layer], self.spk[layer], 
							self.syn[layer], self.mem[layer])

				self.record(layer)

		self.stack_rec()


if __name__ == "__main__": 

	from net_dict import net_dict

	spiker = SNN(net_dict)

	spiker.forward(torch.ones((4, 10)))

	print(spiker.mem_rec)
	print(spiker.syn_rec)
	print(spiker.spk_rec)
