#!/Users/alessio/anaconda3/bin/python3

# Script which simulates the model developed for the layer 

from layerFunctions import createSparseArray
from snnFunctions import simulateSnn, \
			createNetworkDictList, \
			plotNetworkResults



import numpy as np

# Number of simulation cycles
N_sim = 1000

# List describing the network. Each entry corresponds to the number of nodes in the
# specific layer
networkList = [1,1]

# Density of the input spikes
density = 0.1

# Thresholds
v_th_list = [50*np.ones(networkList[i]) for i in range(1,len(networkList))]

# Membrane potential reset state
v_reset = 0 

# Range of the weights to generate
w_min_list = [10*np.ones(networkList[i]) for i in range(1,len(networkList))]

w_max_list = [40*np.ones(networkList[i]) for i in range(1,len(networkList))]

dt_tau = 0.3

# Create the network dictionary list
networkDictList = createNetworkDictList(v_th_list, v_reset, w_min_list, w_max_list, networkList)

# Create the bidimensional array containing the input events
inEvents_evolution = createSparseArray(N_sim, networkList[0], density)

# Create the output events array
outEventsEvol_list = [np.zeros((N_sim, networkList[i])).astype(bool) for i 
			in range(1, len(networkList))]

# Create the array of membrane potentials
v_memEvol_list = [np.zeros((N_sim, networkList[i])) for i in range(1, len(networkList))]

# Simulate the neuron
simulateSnn(N_sim, inEvents_evolution, networkDictList, outEventsEvol_list, 
		v_memEvol_list, dt_tau, v_reset)

# Transpose the arrays in order to plot them with respect to time
inEvents_evolution = inEvents_evolution.T


plotNetworkResults(networkList, inEvents_evolution, outEventsEvol_list, 
			v_memEvol_list, v_th_list)
