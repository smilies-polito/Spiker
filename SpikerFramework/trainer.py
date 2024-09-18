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


	def evaluate(self, dataloader):

		# Test set
		with torch.no_grad():

			self.net.eval()

			# Iterate over the dataloader
			for _, (data, _, labels) in enumerate(dataloader):

				data 	= data.to(self.device)
				labels	= labels.to(self.device)

				_, out_rec = net(test_data)

				# Reshape mem_rec to combine the time and batch dims
				out_rec_flat = out_rec.view(-1, out_rec.shape[-1])
				labels_repeat = test_labels.repeat(out_rec.shape[0])

				# Compute the loss over all time steps at once
				loss_val = loss_fn(out_rec_flat, labels_repeat)

		_, idx = out_rec.sum(dim=0).max(1)
		accuracy = np.mean((labels == idx).detach().cpu().numpy())

		return loss_val, accuracy


	def log(self, epoch, train_loss, val_loss, train_acc, val_acc, start_time =
			None):

		epoch = str(epoch)

		if start_time:
			elapsed = time.time() - start_time
			elapsed = "{:.2f}".format(elapsed) + "s"

		train_loss = "{:.2f}".format(train_loss)
		val_loss = "{:.2f}".format(val_loss)

		train_acc = str(train_acc * 100) + "%"
		val_acc = str(val_acc * 100) + "%"

		logging.info("Epoch " + epoch)
		logging.info("Elapsed time: " + elapsed)
		logging.info("Train loss: ", train_loss)
		logging.info("Train loss: ", val_loss)
		logging.info("Trining accuracy: ", train_acc)
		logging.info("Trining accuracy: ", val_acc)


	def store(self, out_dir, out_file = None):

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		if not out_file:
			out_file = "trained_state_dict.pt"

		out_path	= out_dir + "/" + out_file

		torch.save(self.net.state_dict(), out_path)
