import os
import tonic
import torch
from torch.utils.data import DataLoader

class ShdDL:

	def __init__(self, data_dir, transform = "default", download = True,
			num_steps = 100):

		if transform == "default":
			self.transform = tonic.transforms.Compose([
				tonic.transforms.ToFrame(
					sensor_size = tonic.datasets.hsd.SHD.sensor_size,
					n_time_bins = num_steps
				),
				Squeeze(dim = 1),
				ToFloatTensor()
			])
				

		else:
			self.transform = transform

		self.num_cpu_cores = os.cpu_count()

		self.train_set = tonic.datasets.hsd.SHD(
				save_to		= data_dir,
				train		= True,
				transform	= self.transform
		)

		self.test_set = tonic.datasets.hsd.SHD(
				save_to		= data_dir,
				train		= False,
				transform	= self.transform
		)


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

class Squeeze:

	def __init__(self, dim):

		self.dim = dim

	def __call__(self, tensor):

		if tensor.shape[self.dim] == 1:
			return tensor.squeeze(self.dim)

		raise ValueError(f"Expected size 1 at dimension {self.dim}, but got"
				f"size {tensor.shape[self.dim]}.")

class ToFloatTensor:
    def __call__(self, array):
        return torch.tensor(array, dtype=torch.float32)

