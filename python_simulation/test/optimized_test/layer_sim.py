#!/Users/alessio/anaconda3/bin/python3

# Script which simulates the model developed for the neuron 

from neuronFunctions import createSparseArray, 	\
			createArray

from layerFunctions import simulateLayer,	\
			createDictList,		\
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
v_res = np.zeros(currLayerDim)

# Range of the weights to generate
w_min = 2*np.ones(currLayerDim)	
w_max = 100*np.ones(currLayerDim)

dt_tau = 0.3

# Create the neuron dictionary
neuronsDictList = createDictList(currLayerDim, v_th, v_res, prevLayerDim, w_min, w_max)

# Create the bidimensional array containing the input events
inEvents_evolution = createSparseArray(N_sim, prevLayerDim, density)

# Create the output events array
outEvents_evolution = createArray(N_sim, currLayerDim)

# Create the array of membrane potentials
v_mem_evolution = createArray(N_sim, currLayerDim)

# Simulate the neuron
simulateLayer(N_sim, inEvents_evolution, neuronsDictList, dt_tau, outEvents_evolution, 
		v_mem_evolution, currLayerDim)


# Transpose the arrays of input events in order to plot it
# with respect to time
inEvents_evolution = inEvents_evolution.T
outEvents_evolution = outEvents_evolution.T
v_mem_evolution = v_mem_evolution.T

# Plot the results
plotLayerResults(currLayerDim, prevLayerDim, inEvents_evolution, outEvents_evolution, 
			v_mem_evolution, v_th)

