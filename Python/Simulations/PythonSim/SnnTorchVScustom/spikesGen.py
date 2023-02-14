import snntorch as snn
from snntorch import spikegen

import torch
import torch.nn as nn

from mnist import loadDataset
from utils import createDir

from runParameters import *
from files import *

createDir(spikesDir)

# Import dataset
train_loader, test_loader = loadDataset(data_path, batch_size)
test_batch = iter(test_loader)

# Loop over the batches
for test_data, test_targets in test_batch:

	# Generate spikes
	input_spikes = spikegen.rate(test_data, num_steps = num_steps, gain = 1)

	# Store spikes on file
	with open(spikesFilename, "wb") as fp:
		torch.save(input_spikes, fp)

	sp.run(custom, shell = True, executable = "/bin/bash")
	sp.run(snntorch_model, shell = True, executable = "/bin/bash")
