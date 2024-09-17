import re
import torch.nn as nn
import snntorch as snn


class SNN(nn.Module):

	def __init__(self, net_dict):

		super(SNN, self).__init__()

		self.layers = nn.ModuleDict()

		self.mem = {}
		self.spk = {}

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

	def reset_snn(self):

		for layer in self.layers:

			idx = str(self.extract_index(layer))

			if "fc" not in layer:
				if layer == "if" + idx:
					self.layers[layer].


				
	def extract_index(self, layer_name):

		index = re.findall(r'\d+', layer_name)

		if len(index) != 1:

			error_msg = "Invalid layer name: " + str(layer_name)
			error_msg += ". Use \"layer_\" + <integer layer index>\n"

			raise ValueError(error_msg)
		else:
			return int(index[0])



	def forward(self, input_spikes):

		# Initialize hidden states at t=0
		mem1 = self.lif1.reset_mem()
		mem_out = self.readout.reset_mem()

		# Record the final layer
		mem_out_rec = []
		out_rec = []

		for step in range(input_spikes.shape[0]):

			cur1 = self.fc1(input_spikes[step])
			spk1, mem1 = self.lif1(cur1, mem1)

			cur2 = self.fc2(spk1)
			mem_out, out = self.readout(cur2, mem_out)

			mem_out_rec.append(mem_out)
			out_rec.append(out)

		return torch.stack(mem_out_rec, dim=0), \
			torch.stack(out_rec, dim=0)

if __name__ == "__main__": 

	from net_dict import net_dict

	spiker = SNN(net_dict)

	print(spiker)
