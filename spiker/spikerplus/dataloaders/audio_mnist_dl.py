import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as fn

class AudioMnistDL:

	def __init__(self, data_dir,
		fft_window 		= 25e-3, #s
		hop_length_s	= 10e-3, #s
		n_channels		= 40,
		spiking_thresh	= 0.9,
		transform		= "default",
		train_size 		= 0.8
	):

		# Input data sample rate
		self.sample_rate		= 48e3 # Hz

		# Short Term Fourier Transform (STFT) window
		self.fft_window			= fft_window

		# Step from one window to the other (controls overlap)
		self.hop_length_s		= hop_length_s

		# Number of input channels: filters in the mel bank
		self.n_mels				= n_channels

		# Spiking threshold
		self.spiking_thresh 	= spiking_thresh

		if transform == "default":

			self.transform = MelFilterbank(
				sample_rate 		= self.sample_rate,
				fft_window 			= self.fft_window,
				hop_length_s		= self.hop_length_s,
				n_mels 				= self.n_mels,
				db 					= True,
				normalize			= True,
				spikify				= True,
				spiking_thresh		= self.spiking_thresh
			)

		else:
			self.transform = transform

		self.dataset = CustomDataset(
			root_dir	= data_dir,
			transform	= self.transform
		)

		self.num_cpu_cores = os.cpu_count()

		# Train/test split
		train_len 		= int(train_size * len(self.dataset))
		test_len		= len(self.dataset) - train_len

		# Split the dataset into training and validation sets
		self.train_set, self.test_set = random_split(self.dataset,
				[train_len, test_len])


	def load(self, train_drop_last = True, train_shuffle =
			True, test_drop_last = True, test_shuffle = True, batch_size = 64,
			num_workers = None):

		if not num_workers:
			num_workers = self.num_cpu_cores

		train_loader = DataLoader(self.train_set, 
			batch_size	= batch_size,
			shuffle		= train_shuffle,
			num_workers	= num_workers,
			drop_last 	= train_drop_last
		)

		test_loader = DataLoader(self.test_set, 
			batch_size	= batch_size,
			shuffle		= test_shuffle,
			num_workers	= num_workers,
			drop_last 	= test_drop_last
		)

		return train_loader, test_loader


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

		return waveform, label


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
