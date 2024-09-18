import time

class Trainer:

	def __init__(self, net, optimizer = None, loss_fn = None):

		self.net = net

		if not optimizer:

			adam_beta1	= 0.9
			adam_beta2	= 0.999
			lr			= 5e-4

			self.optimizer = torch.optim.Adam(
					self.net.parameters(),
					lr		= lr,
					betas	= (adam_beta1, adam_beta2)
			)

		else:

			self.optimizer = optimizer


		if not loss_fn:
			loss_fn = nn.CrossEntropyLoss()

		else:
			self.loss_fn = loss_fn

		if torch.cuda.is_available():
			self.device = torch.device("cuda")

		else:
			self.device = torch.device("cpu")

		self.net.to(self.device)
