import os
import torch
import torchaudio
from torch.utils.data import Dataset

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
