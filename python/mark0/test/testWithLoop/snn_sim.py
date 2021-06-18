#!/Users/alessio/anaconda3/bin/python3

# Script which simulates an entire spiking neural network and plots the temporal evolution
# of the spikes and the membrane potentials 

from neuronFunctions import createSparseArray

from snnFunctions import simulateSnn, \
			createNetworkDictList, \
			createArraysList, \
			plotNetworkResults

import numpy as np

# Number of simulation cycles
N_sim = 1000

# List which describes the network. Each entry represents the number of neurons in the
# corresponding layer
layersList = [4, 3, 2]

# Density of the input spikes
density = 0.01

# Thresholds
v_th_list = [50*np.ones(layersList[i]) for i in range(1, len(layersList))]

# Membrane potential reset state
v_res_list = [np.zeros(layersList[i]) for i in range(1, len(layersList))]


# Range of the weights to generate
w_min_list = [2*np.ones(layersList[i]) for i in range(1, len(layersList))]	
w_max_list = [60*np.ones(layersList[i]) for i in range(1, len(layersList))]

dt_tau = 0.3

# Create the neuron dictionaries
networkDictList = createNetworkDictList(layersList, v_th_list, v_res_list, w_min_list, 
			w_max_list)


# Create the bidimensional array containing the input events
inEvents_evolution = createSparseArray(N_sim, layersList[0], density)

# Create the output events array
outEventsEvol_list = [np.zeros((N_sim, layersList[i])).astype(bool) 
			for i in range(1, len(layersList))] 


# Create the array of membrane potentials
v_memEvol_list = createArraysList(layersList, N_sim)


# Simulate the network
simulateSnn(N_sim, inEvents_evolution, networkDictList, dt_tau, outEventsEvol_list,
		v_memEvol_list, layersList)


# Transpose the arrays of input events in order to plot it
# with respect to time
inEvents_evolution = inEvents_evolution.T


# Plot the results
plotNetworkResults(layersList, inEvents_evolution, outEventsEvol_list, 
			v_memEvol_list, v_th_list)
