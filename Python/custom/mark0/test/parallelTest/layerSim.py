#!/Users/alessio/anaconda3/bin/python3

# Script which simulates the model developed for the layer 

from layerFunctions import createSparseArray, \
			simulateLayer,	\
			createLayerDict, \
			plotLayerResults

import numpy as np

# Number of simulation cycles
N_sim = 1000

# Number of neurons in the previous and current layers
prevLayerDim = 3
currLayerDim = 4

# Density of the input spikes
density = 0.01

# Thresholds
v_th = 50*np.ones(currLayerDim)

# Membrane potential reset state
v_reset = 0

# Range of the weights to generate
w_min = 2*np.ones(currLayerDim)	
w_max = 100*np.ones(currLayerDim)

dt_tau = 0.3

# Create the neuron dictionary
layerDict = createLayerDict(v_th, v_reset, w_min, w_max, currLayerDim, prevLayerDim)

# Create the bidimensional array containing the input events
inEvents_evolution = createSparseArray(N_sim, prevLayerDim, density)

# Create the output events array
outEvents_evolution = np.zeros((N_sim, currLayerDim)).astype(bool) 

# Create the array of membrane potentials
v_mem_evolution = np.zeros((N_sim, currLayerDim))

# Simulate the neuron
simulateLayer(N_sim, inEvents_evolution, layerDict,  outEvents_evolution, 
		v_mem_evolution, dt_tau, v_reset)


# Transpose the arrays in order to plot them with respect to time
inEvents_evolution = inEvents_evolution.T
outEvents_evolution = outEvents_evolution.T
v_mem_evolution = v_mem_evolution.T

# Plot the results
plotLayerResults(currLayerDim, prevLayerDim, inEvents_evolution, outEvents_evolution, 
			v_mem_evolution, v_th)

