#!/Users/alessio/anaconda3/bin/python3

# Script that simulates the behaviour of the developed
# single cycle neuron model

import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt 
import sys



# Add the path containing the script to simulate to the modules
# search path and then import the script
development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/singleCycle"


if development not in sys.path:
	sys.path.insert(1,development)

from neuronSingleCycle import neuronSingleCycle




# Random generator
randGen = np.random.default_rng()

# Store the partial values to plot them
v_mem_values = []
outEvent_values = []

# Initialize the parameters of the neuron
v_mem = 0
v_th_max = 70
v_th_min = 1
dt_tau = 0.1

# Generate a sparse train of input spikes
inEvent = random(1, 1000, density = 0.02, random_state = randGen)
inEvent = inEvent.A[0].astype(bool).astype(int)

# Generate random weights to simulate the neuron
w_min = 2	
w_max = 100
weight = (w_max - w_min)*randGen.\
	random(size=inEvent.size) + w_min


# Simulate the neuron
for i in range(inEvent.size):

	values = neuronSingleCycle(inEvent[i], v_mem, v_th_max, v_th_min, weight[i], dt_tau)
	v_mem = values[0]
	outEvent = values[1]

	v_mem_values.append(v_mem)
	outEvent_values.append(outEvent)


v_th_max = v_th_max * np.ones(len(v_mem_values))

# Plot the obtained results
fig, (plot1, plot2, plot3) = plt.subplots(3, 1)

# Input spikes
plot1.plot(inEvent)
plot1.grid()
plot1.set_xticks(np.arange(0, inEvent.size, step = inEvent.size/20))
plot1.set_title("Input spikes")

# Membrane potential
plot2.plot(v_mem_values)
plot2.plot(v_th_max, "--")
plot2.grid()
plot2.set_xticks(np.arange(0, len(v_mem_values), step = len(v_mem_values)/20))
plot2.set_title("Membrane potential")

# Output spikes
plot3.plot(outEvent_values)
plot3.grid()
plot3.set_xticks(np.arange(0, len(outEvent_values), step = len(outEvent_values)/20))
plot3.set_title("Output spikes")

plt.subplots_adjust(hspace = 0.6)

plt.show()
