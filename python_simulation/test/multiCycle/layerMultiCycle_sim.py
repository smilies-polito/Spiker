#!/Users/alessio/anaconda3/bin/python3

# Script that simulates a layer of multi cycle neurons

import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt
import sys


# Add the path containing the script to simulate to the modules
# search path and then import the script
development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/multiCycle"

if development not in sys.path:
	sys.path.insert(1,development)

from layerMultiCycle import layerMultiCycle





# Number of simulation cycles
N_sim = 1000

# Number of neurons in the previous layer
N_prev = 5

# Number of neurons in the current layer
N_curr = 3

# Random generator
randGen = np.random.default_rng()

# Thresholds
v_th_max = 50*np.ones(N_curr)
v_th_min = 2*np.ones(N_curr)

# Membrane potential
v_mem = np.zeros(N_curr)

# Output arrays
v_mem_evolution = np.zeros((N_sim, N_curr))
outEvents = np.zeros((N_sim, N_curr))

# Generate random weights to simulate the neuron
w_min = 2
w_max = 100
weights = (w_max - w_min)*randGen.\
	random(size=(N_curr, N_prev)) + w_min

# Generate a sparse train of input spikes
inEvents = random(N_sim, N_prev, density = 0.01, 
		random_state = randGen)
inEvents = inEvents.A.astype(bool).astype(int)

dt_tau = 0.3

for i in range(N_sim):

	# Update the layer
	layerMultiCycle(inEvents[i], v_mem, v_th_max, v_th_min, weights, 
		dt_tau, N_curr, N_prev, outEvents[i])

	v_mem_evolution[i] = v_mem


# Transpose the array of input events in order to plot it
# with respect to time
inEvents = inEvents.T
outEvents = outEvents.T
v_mem_evolution = v_mem_evolution.T

# Create the v_th array in order to be able tp plot it
v_th_max = v_th_max * np.ones((N_sim, N_curr))
v_th_max = v_th_max.T


for i in range(N_curr):

	# Plot the obtained results
	fig, axs = plt.subplots(N_prev+2, 1)

	# Input spikes
	for j in range(N_prev):
		axs[j].plot(inEvents[j])
		axs[j].grid()
		axs[j].set_xticks(np.arange(0, inEvents[j].size,\
				step = inEvents[j].size/20))
		axs[j].set_title("Input spikes")

	# Membrane potential
	axs[N_prev].plot(v_mem_evolution[i])
	axs[N_prev].plot(v_th_max[i], "--")
	axs[N_prev].grid()
	axs[N_prev].set_xticks(np.arange(0, v_mem.size,\
				step = v_mem.size/20))
	axs[N_prev].set_title("Membrane potential")

	# Output spikes
	axs[N_prev+1].plot(outEvents[i])
	axs[N_prev+1].grid()
	axs[N_prev+1].set_xticks(np.arange(0, outEvents[i].size, \
				step = outEvents[i].size/20))
	axs[N_prev+1].set_title("Output spikes")

	plt.subplots_adjust(hspace = 0.6)

	plt.show()
