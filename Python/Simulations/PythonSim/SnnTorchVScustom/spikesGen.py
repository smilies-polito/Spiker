import snntorch as snn
from snntorch import spikegen

import torch
import torch.nn as nn

from runParameters import *
from files import *


input_spikes = spikegen.rate(data_it, num_steps = num_steps, gain = 1)
