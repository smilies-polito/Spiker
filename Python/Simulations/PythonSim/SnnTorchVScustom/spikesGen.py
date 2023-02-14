import snntorch as snn
from snntorch import spikegen

import torch
import torch.nn as nn

from mnist import loadDataset
from utils import createDir

from runParameters import *
from files import *

createDir(spikesDir)

train_loader, test_loader = loadDataset(data_path, batch_size)

test_batch = iter(test_loader)

for test_data, test_targets in test_batch:

	input_spikes = spikegen.rate(test_data, num_steps = num_steps, gain = 1)

	with open(spikesFilename, "wb") as fp:
		torch.save(input_spikes, fp)
