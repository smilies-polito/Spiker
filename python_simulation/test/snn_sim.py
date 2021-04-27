#!/Users/alessio/anaconda3/bin/python3

# Script that simulates a layer of multi cycle neurons

import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt
import sys


# Add the path containing the script to simulate to the modules
# search path and then import the script
development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development"

if development not in sys.path:
	sys.path.insert(1,development)

from snn import snn





# Number of simulation cycles
N_sim = 1000

# Network shape = list of layers
layersList = [2, 4, 1]

# Random generator
randGen = np.random.default_rng()

# Thresholds
v_th_max = [50*np.ones(i) for i in layersList[1:]]

v_th_min = [2*np.ones(i) for i in layersList[1:]]

# Membrane potential
v_mem = [np.zeros(i) for i in layersList[1:]]

# Output events
outEvents = [np.zeros(i) for i in layersList[1:]]


# Output arrays
v_mem_evolution = [np.zeros((N_sim, i)) for i in layersList[1:]]
outEvents_evolution = [np.zeros((N_sim, i)) for i in layersList[1:]]

# Generate random weights to simulate the neuron
w_min = 2
w_max = 100
weights = [(w_max - w_min)*randGen.\
	random(size=(layersList[i], layersList[i-1])) + w_min for i\
	in range(1,len(layersList))]


# Generate a sparse train of input spikes
inEvents = random(N_sim, layersList[0], density = 0.01, 
		random_state = randGen)
inEvents = inEvents.A.astype(bool).astype(int)

dt_tau = 0.3

for i in range(N_sim):

	# Update the network
	snn(inEvents[i], layersList[1:], v_mem, v_th_max, v_th_min, 
	weights, dt_tau, outEvents)

	for j in range(len(layersList)-1):
		v_mem_evolution[j][i] = v_mem[j]
		outEvents_evolution[j][i] = outEvents[j]


# Transpose the array of input events in order to plot it
# with respect to time
inEvents = inEvents.T

for i in range(len(layersList)-1):
	outEvents_evolution[i] = outEvents_evolution[i].T
	v_mem_evolution[i] = v_mem_evolution[i].T


N_prev = layersList[0]
outEvents_evolution.insert(0, inEvents)

for i in range(1, len(layersList)):

	for j in range(layersList[i]):

		v_th = v_th_max[i-1][j]

		# Plot the obtained results
		fig, axs = plt.subplots(layersList[i-1]+2, 1)
	
		# Input spikes
		for k in range(layersList[i-1]):

			axs[k].plot(outEvents_evolution[i-1][k])
			axs[k].grid()
			axs[k].set_xticks(np.arange(0, outEvents_evolution[i-1][k].size,\
					step = outEvents_evolution[i-1][k].size/20))
			axs[k].set_title("Input spikes")
	
		# Membrane potential
		v_th = v_th*np.ones(v_mem_evolution[i-1][j].size)
		axs[layersList[i-1]].plot(v_mem_evolution[i-1][j])
		axs[layersList[i-1]].plot(v_th, "--")
		axs[layersList[i-1]].grid()
		axs[layersList[i-1]].set_xticks(np.arange(0, v_mem_evolution[i-1][j].size,\
					step = v_mem_evolution[i-1][j].size/20))
		axs[layersList[i-1]].set_title("Membrane potential")
	
		# Output spikes
		axs[layersList[i-1]+1].plot(outEvents_evolution[i][j])
		axs[layersList[i-1]+1].grid()
		axs[layersList[i-1]+1].set_xticks(np.arange(0, outEvents_evolution[i][j].size, \
					step = outEvents_evolution[i][j].size/20))
		axs[layersList[i-1]+1].set_title("Output spikes")
	
		plt.subplots_adjust(hspace = 2)

		name = "plots/optimizedVersion/layer_" + str(i) + "_node_" + str(j) + ".png"
		plt.savefig(name)
