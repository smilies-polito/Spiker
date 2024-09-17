import re
import torch.nn as nn
import snntorch as snn


class SNN(nn.Module):

	def __init__(self, num_inputs = 40, num_hidden = 128,
			num_outputs = 10, model = "lif", bias = False,
			beta = 0.9, threshold = 1.):

		super(SNN, self).__init__()

		self.layers = nn.ModuleDict()

		self.mem_rec = []
		self.spk_rec = []

	
	def build_network(self, net_dict):

		for key in net_dict:
 
			if "layer" in key:

				idx = str(self.extract_index(key) + 1)

				self.layers["fc" + idx] = nn.Linear(
					in_features	= 3,
					out_features	= 2,
					bias 		= False,
				)


				print(self.layers)
 
 
				# if net_dict[key]["neuron_model"] == "if":
 				# elif net_dict[key]["neuron_model"] == "lif":
 				# elif net_dict[key]["neuron_model"] == "syn":
 				# elif net_dict[key]["neuron_model"] == "rif":
 				# elif net_dict[key]["neuron_model"] == "rlif":
 				# elif net_dict[key]["neuron_model"] == "rsyn":
 				# else:
 

				
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

	snn = SNN()

	print(snn.build_network(net_dict))
