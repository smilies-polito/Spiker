#!/Users/alessio/anaconda3/bin/python3

# Script which simulates the model developed for the layer 

from layerFunctions import createSparseArray, \
			simulateTrainLayer,	\
			createLayerDict, \
			plotTrainLayerResults

import numpy as np

# Number of simulation cycles
N_sim = 1000

# Number of neurons in the previous and current layers
prevLayerDim = 4
currLayerDim = 1

# Density of the input spikes
density = 0.01

# Thresholds
v_th = 50*np.ones(currLayerDim)

# Membrane potential reset state
v_reset = 0

# Range of the weights to generate
w_min = 25*np.ones(currLayerDim)	
w_max = 25*np.ones(currLayerDim)

v_mem_dt_tau = 0.1
stdp_dt_tau = 0.03

A_ltp = 1
A_ltd = -1

# Create the neuron dictionary
layerDict = createLayerDict(v_th, v_reset, w_min, w_max, currLayerDim, prevLayerDim)


# Create the bidimensional array containing the input events
inEvents_evolution = createSparseArray(N_sim, prevLayerDim, density)


# Create the output events array
outEvents_evolution = np.zeros((N_sim, currLayerDim)).astype(bool) 

# Create the array of membrane potentials
v_mem_evolution = np.zeros((N_sim, currLayerDim))

# Create the tridimensional array for the weights
weights_evolution = np.zeros((N_sim, currLayerDim, prevLayerDim))

# Simulate the neuron
simulateTrainLayer(N_sim, inEvents_evolution, layerDict, weights_evolution, 
			outEvents_evolution, v_mem_evolution, v_mem_dt_tau, 
			stdp_dt_tau, v_reset, A_ltp, A_ltd)


# Transpose the arrays in order to plot them with respect to time
inEvents_evolution = inEvents_evolution.T
outEvents_evolution = outEvents_evolution.T
v_mem_evolution = v_mem_evolution.T
weights_evolution = np.transpose(weights_evolution, (1,2,0))

# Plot the results
plotTrainLayerResults(currLayerDim, prevLayerDim, inEvents_evolution, weights_evolution,
			outEvents_evolution, v_mem_evolution, v_th)

