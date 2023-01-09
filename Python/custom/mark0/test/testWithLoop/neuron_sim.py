#!/Users/alessio/anaconda3/bin/python3

# Script which simulates the model developed for the neuron 

from neuronFunctions import createNeuronDict, \
			createSparseArray, \
			createArray, \
			simulateNeuron, \
			plotResults

# Number of simulation cycles
N_sim = 1000

# Number of neurons in the previous and current layers
prevLayerDim = 3
currLayerDim = 1

# Density of the input spikes
density = 0.01

# Thresholds
v_th = 50

# Membrane potential reset state
v_res = 0

# Range of the weights to generate
w_min = 2	
w_max = 100


dt_tau = 0.3

# Create the neuron dictionary
neuronDict = createNeuronDict(v_th, v_res, prevLayerDim, w_min, w_max)

# Create the bidimensional array containing the input events
inEvents = createSparseArray(N_sim, prevLayerDim, density)

# Create the output events array
outEvents = createArray(N_sim, currLayerDim)

# Create the array of membrane potentials
v_mem_evolution = createArray(N_sim, currLayerDim)

# Simulate the neuron
simulateNeuron(N_sim, inEvents, outEvents, v_mem_evolution, neuronDict, dt_tau)


# Transpose the array of input events in order to plot it
# with respect to time
inEvents = inEvents.T

# Plot the results
plotResults(prevLayerDim, inEvents, outEvents, v_mem_evolution, v_th)

