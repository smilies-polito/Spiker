import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as fn

class Trainer:

	def __init__(self, net, readout_type = "mem", optimizer = None,
			loss_fn = None):

		self.supported_readouts = [
			"spk",
			"spk_count",
			"mem",
			"mem_softmax",
			"mem_max",
			"mem_avg"
		]

		self.net			= net

		if readout_type in self.supported_readouts:
			self.readout_type	= readout_type

		else:
			raise ValueError("Invalid readout type. Choose between " +
					str(self.supported_readouts) + "\n")

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
			self.loss_fn = nn.CrossEntropyLoss()

		else:
			self.loss_fn = loss_fn

		if torch.cuda.is_available():
			self.device = torch.device("cuda")

		else:
			self.device = torch.device("cpu")

		self.net.to(self.device)

		log_message = "Training set-up: \n"
		log_message += "Readout type: " + str(self.readout_type) + "\n"
		log_message += "Optimizer: " + str(self.optimizer) + "\n"
		log_message += "Loss function: " + str(self.loss_fn) + "\n"
		log_message += "Device: " + str(self.device) + "\n"

		logging.info(log_message)



	def train(self, train_loader, val_loader, n_epochs = 20, store = False,
			output_dir = "Trained"):

		train_loss = torch.zeros(n_epochs)
		val_loss = torch.zeros(n_epochs)

		logging.info("Begin training\n")
		start_time = time.time()

		for epoch in range(n_epochs):

			train_loss, train_acc = self.train_one_epoch(train_loader)
			val_loss, val_acc = self.evaluate(val_loader)

			self.log(epoch, train_loss, val_loss, train_acc, val_acc,
					start_time)

		if store:
			self.store(output_dir)



	def train_one_epoch(self, dataloader):

		# Iterate over the dataloader
		for batch_idx, (data, labels) in enumerate(dataloader):

			data 	= data.permute(1, 0, 2).to(self.device)
			labels	= labels.to(self.device)

			self.optimizer.zero_grad()

			self.net.train()

			self.net(data)

			out_rec, targets = self.readout(labels)

			# Compute the loss over all time steps at once
			loss_val = self.loss_fn(out_rec, targets)

			# Gradient calculation + weight update
			loss_val.backward()
			self.optimizer.step()

		accuracy = self.compute_accuracy(labels)

		return loss_val.item(), accuracy.item()


	def evaluate(self, dataloader):

		# Test set
		with torch.no_grad():

			self.net.eval()

			# Iterate over the dataloader
			for _, (data, labels) in enumerate(dataloader):

				data 	= data.permute(1, 0, 2).to(self.device)
				labels	= labels.to(self.device)

				self.net(data)

				out_rec, targets = self.readout(labels)

				# Compute the loss over all time steps at once
				loss_val = self.loss_fn(out_rec, targets)

		accuracy = self.compute_accuracy(labels)

		return loss_val.item(), accuracy.item()


	def readout(self, labels):

		if "mem" in self.readout_type:
			_, out_rec = list(self.net.mem_rec.items())[-1]

			if self.readout_type == "mem_max":
				out_rec, _ = torch.max(out_rec, dim = 0, keepdim = True)

			elif self.readout_type == "mem_avg":
				out_rec = torch.mean(out_rec, dim = 0, keepdim = True)

			elif self.readout_type == "mem_softmax":
				out_rec = fn.softmax(out_rec, dim = -1)


		else:
			_, out_rec = list(self.net.spk_rec.items())[-1]

			if self.readout_type == "spk_count":
				out_rec = torch.sum(out_rec, dim = 0, keepdim = True)

		return out_rec.reshape(-1, out_rec.shape[-1]), \
				labels.repeat(out_rec.shape[0])


	def compute_accuracy(self, labels):

		if "mem" in self.readout_type:
			_, out_rec = list(self.net.mem_rec.items())[-1]

			_, idx = torch.mean(out_rec, dim = 0).max(dim = 1)

		else:
			_, out_rec = list(self.net.spk_rec.items())[-1]

			_, idx = torch.sum(out_rec, dim = 0).max(dim = 1)

		return torch.mean((labels == idx).float().detach().cpu())



	def log(self, epoch, train_loss, val_loss, train_acc, val_acc, start_time =
			None):

		log_message = ""

		epoch = str(epoch)
		log_message += "Epoch " + epoch + "\n"

		if start_time:
			elapsed = time.time() - start_time
			elapsed = "{:.2f}".format(elapsed) + "s"
			log_message += "Elapsed time " + elapsed + "\n"

		train_loss = "{:.2f}".format(train_loss)
		val_loss = "{:.2f}".format(val_loss)
		log_message += "Train loss: " + train_loss + "\n"
		log_message += "Validation loss: " + val_loss + "\n"

		train_acc = str(train_acc * 100) + "%"
		val_acc = str(val_acc * 100) + "%"
		log_message += "Train accuracy: " + train_acc + "\n"
		log_message += "Validation accuracy: " + val_acc + "\n"

		logging.info(log_message)


	def store(self, out_dir, out_file = None):

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		if not out_file:
			out_file = "trained_state_dict.pt"

		out_path	= out_dir + "/" + out_file

		torch.save(self.net.state_dict(), out_path)


if __name__ == "__main__": 

	from torch.utils.data import DataLoader, random_split

	from net_dict import net_dict
	from net_builder import NetBuilder

	import sys

	audio_mnist_dir = "../../SnnModels/SnnTorch/AudioMnist/"

	if audio_mnist_dir not in sys.path:
		sys.path.append(audio_mnist_dir)

	from audio_mnist import MelFilterbank, CustomDataset

	logging.basicConfig(level=logging.INFO)

	root_dir	= audio_mnist_dir + "/Data"
	batch_size	= 128

	sample_rate	= 48e3

	# Short Term Fourier Transform (STFT) window
	fft_window	= 25e-3 # s

	# Step from one window to the other (controls overlap)
	hop_length_s	= 10e-3 #s

	# Number of input channels: filters in the mel bank
	n_mels		= 40

	# Spiking threshold
	spiking_thresh 	= 0.9

	transform = MelFilterbank(
		sample_rate 	= sample_rate,
		fft_window 	= fft_window,
		hop_length_s	= hop_length_s,
		n_mels 		= n_mels,
		db 		= True,
		normalize	= True,
		spikify		= True,
		spiking_thresh	= spiking_thresh
	)

	dataset = CustomDataset(
		root_dir	= root_dir,
		transform	= transform
	)

	# Train/test split
	train_size 		= int(0.8 * len(dataset))
	test_size		= len(dataset) - train_size

	# Split the dataset into training and validation sets
	train_set, test_set = random_split(dataset, [train_size, test_size])

	train_loader = DataLoader(train_set, 
 		batch_size	= batch_size,
 		shuffle		= True,
 		num_workers	= 4,
 		drop_last 	= True
 	)

	test_loader = DataLoader(test_set, 
 		batch_size	= batch_size,
 		shuffle		= True,
 		num_workers	= 4,
 		drop_last 	= True
 	)


	net_builder		= NetBuilder(net_dict)

	snn = net_builder.build()

	trainer 		= Trainer(snn)

	trainer.train(train_loader, test_loader)
