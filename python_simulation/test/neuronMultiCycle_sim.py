#!/Users/alessio/anaconda3/bin/python3

# Script that simulates the model of the multi cycle neuron

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

from neuronMultiCycle import neuronMultiCycle





# Number of simulation cycles
N_sim = 1000

# Number of neurons in the previous layer
N_prevNeurons = 5

# Random generator
randGen = np.random.default_rng()

# Thresholds
v_th_max = 50
v_th_min = 2

# Membrane potentials
v_mem = 0

# Generate random weights to simulate the neuron
w_min = 2	
w_max = 100
weights = (w_max - w_min)*randGen.\
	random(size=N_prevNeurons) + w_min

# Generate a sparse train of input spikes
inEvents = random(N_sim, N_prevNeurons, density = 0.01, 
		random_state = randGen)
inEvents = inEvents.A.astype(bool).astype(int)

dt_tau = 0.3

# Empty lists that will be filled with the input and output 
# values during the neuron's evolution
v_mem_values = []
outEvents_values = []

for i in range(N_sim):

	# Update the neuron
	outValues = neuronMultiCycle(inEvents[i], v_mem, v_th_max, 
			v_th_min, weights, dt_tau, N_prevNeurons)

	# Update the potential
	v_mem = outValues[0]

	# Add the output of the neuron to the array that will be
	# used for the plot
	v_mem_values = np.append(v_mem_values, v_mem)
	outEvents_values = np.append(outEvents_values, 
				outValues[1])


# Transpose the array of input events in order to plot it
# with respect to time
inEvents = inEvents.T

# Create the v_th array in order to be able tp plot it
v_th_max = v_th_max * np.ones(inEvents[0].size)

# Plot the obtained results
fig, axs = plt.subplots(N_prevNeurons+2, 1)

# Input spikes
for i in range(N_prevNeurons):
	axs[i].plot(inEvents[i])
	axs[i].grid()
	axs[i].set_xticks(np.arange(0, inEvents[i].size, 
			step = inEvents[i].size/20))
	axs[i].set_title("Input spikes")

# Membrane potential
axs[N_prevNeurons].plot(v_mem_values)
axs[N_prevNeurons].plot(v_th_max, "--")
#axs[N_prevNeurons].grid()
axs[N_prevNeurons].set_xticks(np.arange(0, len(v_mem_values), 
			step = len(v_mem_values)/20))
axs[N_prevNeurons].set_title("Membrane potential")

# Output spikes
axs[N_prevNeurons+1].plot(outEvents_values)
axs[N_prevNeurons+1].grid()
axs[N_prevNeurons+1].set_xticks(np.arange(0, len(outEvents_values), 
			step = len(outEvents_values)/20))
axs[N_prevNeurons+1].set_title("Output spikes")

plt.subplots_adjust(hspace = 0.6)

plt.show()
