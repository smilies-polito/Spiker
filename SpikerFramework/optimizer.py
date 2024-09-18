import logging
import json

class Optimizer:

	def __init__(self, net, optim_config):

		self.optim_config = self.parse_config(optim_config)


		

	def parse_config(self, optim_config):

		parsed_dict = {}

		log_message = "Optimizer configured: \n"
		log_message += json.dumps(parsed_dict, indent = 4)
		logging.info(log_message)

		return parsed_dict

if __name__ == "__main__":

	from optim_config import optim_config

	logging.basicConfig(level=logging.INFO)

	opt = Optimizer(None, optim_config)
