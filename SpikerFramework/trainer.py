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


	def train_one_epoch(self, dataloader):

		# Iterate over the dataloader
		for batch_idx, (data, _, labels) in enumerate(dataloader):

			data 	= data.to(self.device)
			labels	= labels.to(self.device)

			self.optimizer.zero_grad()

			self.net.train()

			_, out_rec = net(data)

			# Reshape mem_rec to combine the time and batch dims
			out_rec_flat = out_rec.view(-1, out_rec.shape[-1])
			labels_repeat = labels.repeat(out_rec.shape[0])

			# Compute the loss over all time steps at once
			loss_val = self.loss_fn(out_rec_flat, labels_repeat)

			# Gradient calculation + weight update
			loss_val.backward()
			self.optimizer.step()

		_, idx = out_rec.sum(dim=0).max(1)
		accuracy = np.mean((labels == idx).detach().cpu().numpy())
