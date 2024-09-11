import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as fn
import snntorch as snn
from snntorch.functional.quant import state_quant as quantize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class CustomDataset(Dataset):

	def __init__(self, root_dir, transform=None, max_length = 35000):

		"""
		Args:
			root_dir	: str. Directory containing
					subdirectories, one for each user

			transform	: callable, optional. Transform to be
					applied on a sample.
		"""
		self.root_dir 	= root_dir
		self.transform	= transform
		self.max_length	= max_length

		self.data = []

		# Loop over all the users' directories
		for user_folder in os.listdir(root_dir):

			user_path = os.path.join(root_dir, user_folder)

			if os.path.isdir(user_path):

				# Loop over all the WAV recordings
				for file_name in os.listdir(user_path):

					if file_name.endswith(".wav"):

						file_path = os.path.join(
							user_path,
							file_name
						)
						# Extract label from filename
						label = int(
							file_name.split("_")[0]
						)
						self.data.append(
							(file_path, label)
						)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		file_path, label = self.data[idx]
		waveform, sample_rate = torchaudio.load(file_path)

		# Pad or truncate the waveform to match max_length
		if waveform.size(1) > self.max_length:
			waveform = waveform[:, :self.max_length]

		elif waveform.size(1) < self.max_length:
			pad_size = self.max_length - waveform.size(1)
			waveform = fn.pad(waveform, (0, pad_size))

		if self.transform:
			waveform = self.transform(waveform)

		# --- If converting to snnTorch the part under this can be
		# modified ---

		# Reshape and return lens to make it compatible with sparch
		waveform = waveform.squeeze(dim=0).permute(1, 0)

		lens = torch.tensor([x.shape[0] for x in waveform])

		return waveform, lens, label

class MelFilterbank:

	def __init__(self, sample_rate = 48e3, fft_window = 25e-3,
		hop_length_s = 10e-3, n_mels = 40, db = False, normalize = False,
		spikify = False, spiking_thresh = 0.9):

		self.sample_rate	= sample_rate
		self.n_fft		= int(fft_window * sample_rate)
		self.hop_length		= int(hop_length_s * sample_rate)
		self.n_mels		= n_mels

		self.db			= db

		if self.db:
			# Convert the Mel Spectrogram to dB scale
			self.db_transform = torchaudio.transforms.\
						AmplitudeToDB()

		self.normalize		= normalize
		self.spikify		= spikify
		self.spiking_thresh	= spiking_thresh

		# Define the MelSpectrogram transform
		self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
			sample_rate	= self.sample_rate,
			n_fft		= self.n_fft,
			hop_length	= self.hop_length,
			n_mels		= self.n_mels
		)


	def __call__(self, waveform):

		# Apply the Mel Spectrogram transform
		mel_spec = self.mel_spectrogram(waveform)

		if self.db:
			# Convert the Mel Spectrogram to dB scale
			mel_spec = self.db_transform(mel_spec)

		if self.normalize:
			# Normalize mel spectrogram
			mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

		if self.spikify:
			# Convert spectrogram into spike trains
			mel_spec = (mel_spec > self.spiking_thresh).float()

		return mel_spec


class Readout(snn.Leaky):

	def __init__(self, beta):

		super().__init__(beta, reset_mechanism = "none")

	def forward(self, cur, mem):

		_, mem = super().forward(cur, mem)

		return mem, fn.softmax(mem, dim = 1)


class Net(nn.Module):

	def __init__(self, num_inputs = 40, num_hidden = 128,
			num_outputs = 10, model = "lif", bias = False,
			beta = 0.9, threshold = 1., state_quant = False):

		super().__init__()

		self.fc1 = nn.Linear(
			in_features	= num_inputs,
			out_features	= num_hidden,
			bias 		= bias
		)

		self.lif1 = snn.Leaky(
			beta		= beta,
			threshold 	= threshold,
			reset_mechanism	= "subtract",
			state_quant	= state_quant
		)

		self.fc2 = nn.Linear(
			in_features	= num_hidden,
			out_features	= num_outputs,
			bias 		= bias
		)

		self.readout = Readout(
			beta		= beta
		)

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


def print_batch_accuracy(net, data, batch_size, targets, train = False):

	_, output = net(data)
	_, idx = output.sum(dim=0).max(1)

	acc = np.mean((targets == idx).detach().cpu().numpy())

	if train:
		print("Train set accuracy for a single minibatch: " +  
			str(acc*100) + "%")
	else:
		print(f"Test set accuracy for a single minibatch: " + 
			str(acc*100) + "%")


def train_printer(net, batch_size, epoch, iter_counter, loss_hist,
		test_loss_hist, counter, data, targets, test_data,
		test_targets):

	print(f"Epoch {epoch}, Iteration {iter_counter}")

	print(f"Train Set Loss: {loss_hist[counter]:.2f}")

	print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")

	print_batch_accuracy(net, data, batch_size, targets, train = True)

	print_batch_accuracy(net, test_data, batch_size, test_targets, 
			train = False)

	print("\n")



if __name__ == "__main__": 

	import matplotlib.pyplot as plt
	import sounddevice as sd
	import time

	start_time = time.time()

	root_dir	= "../../Models/SnnTorch/AudioMnist/Data"
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

	print("Importing the dataset\n")

	dataset = CustomDataset(
		root_dir	= root_dir,
		transform	= transform
	)

	# Train/test split
	train_size 		= int(0.8 * len(dataset))
	test_size		= len(dataset) - train_size

	print("Splitting into train and test\n")

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

	# Network Architecture
	n_epochs		= 20

	num_inputs		= n_mels
	num_hidden		= 150
	num_outputs		= 10

	w_bits			= 6
	mem_bits		= 8

	beta			= 0.9375

	trained_param_dir	= "../../Models/SnnTorch/AudioMnist/TrainedParameters"

	if not os.path.exists(trained_param_dir):
		os.makedirs(trained_param_dir)

	trained_param_dict	= trained_param_dir + "/state_dict_audiomnist.pt"

	# Optimizer
	adam_beta1	= 0.9
	adam_beta2	= 0.999
	lr		= 5e-4

	mem_quant = quantize(num_bits = mem_bits)

	print("Creating the network\n")

	net = Net(
		num_inputs	= num_inputs,
		num_hidden	= num_hidden,
		num_outputs	= num_outputs,
		beta		= beta,
		state_quant	= mem_quant
	)

	state_dict = torch.load(trained_param_dict, map_location = device)

	for key in state_dict.keys():
		if "weight" in key:

			sigma, mean = torch.std_mean(state_dict[key])

			weight_quant = quantize(num_bits = w_bits, threshold =
					2*sigma, upper_limit = 0)

			state_dict[key] = weight_quant(state_dict[key])

	net.load_state_dict(state_dict)

	loss_fn = nn.CrossEntropyLoss()

	test_loss_hist = []
	counter = 0
	iter_counter = 0

	net.to(device)

	print("Starting the testing loop\n")


	# Test set
	with torch.no_grad():

		net.eval()

		acc = 0
		counter = 0

		# Iterate over the dataloader
		for _, (test_data, _, test_labels) in \
			enumerate(test_loader):

			test_data 	= test_data.permute(1, 0, 2).\
					to(device)
			test_labels	= test_labels.to(device)

			_, out_rec = net(test_data)

			_, idx = out_rec.sum(dim=0).max(1)

			acc += np.mean((test_labels == idx).detach().cpu().numpy())
			counter += 1


			print("Accuracy: " + "{:.2f}".format(acc / counter * 100) + "%")


end_time = time.time()

print("Total required time: " + "{:.2f}".format(end_time - start_time) + "s")
