import logging
import json
import torch

class Optimizer:

	def __init__(self, net, optim_config):

		self.default_config = {

			"weights_bw"	: {
				"min"	: 4,
				"max"	: 8
			},

			"neruons_bw"	: {
				"min"	: 4,
				"max"	: 10
			}
		}

		self.allowed_keys = self.default_config.keys()

		self.net = net

		self.optim_config = self.parse_config(optim_config)

		

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


	def optimize(self):


		pass


	def fixed_point(self, value, fp_dec, bitwidth):

		quant = value * 2**fp_dec

		return self.saturated_int(quant, bitwidth))

	def saturated_int(self, value, bitwidth):
		return self.saturate(self.to_int(value), bitwidth)

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



if __name__ == "__main__":

	from optim_config import optim_config
	from net_builder import NetBuilder
	from net_dict import net_dict

	logging.basicConfig(level=logging.INFO)

	net_builder = NetBuilder(net_dict)

	snn = net_builder.build()

	opt = Optimizer(snn, optim_config)
