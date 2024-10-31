import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as fn
from snntorch import spikegen

class MnistDL:

	def __init__(self, data_dir, transform = "default", download = True,
			image_width = 28, image_height = 28, num_steps = 100, gain = 1):

		self.spike_transform = SpikeTransform(
				num_steps	= num_steps,
				gain		= gain
		)

		if transform == "default":
			self.transform = transforms.Compose([
				transforms.Resize((image_width, image_height)),
				transforms.Grayscale(),
				transforms.ToTensor(),
				transforms.Normalize((0,), (1,)),
				self.spike_transform]
			)
		else:
			self.transform = transform

		self.num_cpu_cores = os.cpu_count()

		self.train_set = datasets.MNIST(
				root 		= data_dir,
				train		= True,
				download	= download,
				transform	= self.transform
		)

		self.test_set = datasets.MNIST(
				root		= data_dir,
				train		= False,
				download	= download,
				transform	= self.transform)


	def load(self, train_drop_last = True, train_shuffle = True, test_drop_last
			= True, test_shuffle = True, batch_size = 64, num_workers = None):

		if not num_workers:
			num_workers = self.num_cpu_cores

		train_loader = DataLoader(self.train_set,
				batch_size	= batch_size,
				shuffle		= train_shuffle,
				num_workers	= num_workers,
				drop_last	= train_drop_last
		)

		test_loader = DataLoader(self.test_set,
				batch_size	= batch_size,
				shuffle		= test_shuffle,
				num_workers	= num_workers,
				drop_last	= test_drop_last)

		return train_loader, test_loader


class SpikeTransform:

	def __init__(self, num_steps = 100, gain = 1):

		self.num_steps	= num_steps
		self.gain		= gain
    
	def __call__(self, img):

		img = img.reshape(img.shape[1]*img.shape[2])

		spikes = spikegen.rate(img,
			num_steps	= self.num_steps,
			gain 		= self.gain
		)

		return spikes
