#!/Users/alessio/anaconda3/bin/python3

# Script which simulates the model developed for the layer 

from snnDataStruct import createNetworkDictList 

from simFunctions import createSparseArray,		\
			plotTrainNetworkResults

from training import trainSingleImg



import numpy as np

# Number of simulation cycles
N_sim = 1000

# List describing the network. Each entry corresponds to the number of nodes in the
# specific layer
networkList = [4,3]

# Density of the input spikes
density = 0.01

# Thresholds
v_th_list = [50*np.ones(networkList[i]) for i in range(1,len(networkList))]

# Membrane potential reset state
v_reset = 0 

# Range of the weights to generate
w_min_list = [10*np.ones(networkList[i]) for i in range(1,len(networkList))]

w_max_list = [40*np.ones(networkList[i]) for i in range(1,len(networkList))]

dt_tau = 0.055

A_ltp = 1
A_ltd = -1

label = 2

labelsArray = np.linspace(0, networkList[-1]-1, networkList[-1]).astype(int)
print("Initial labels array")
print(labelsArray)
print("\n")

spikeCountArray = np.zeros(networkList[-1]).astype(int)

# Create the network dictionary list
networkDictList = createNetworkDictList(v_th_list, v_reset, w_min_list, w_max_list, 
			networkList)

# Create the bidimensional array containing the input events
poissonImg = createSparseArray(N_sim, networkList[0], density)

# Create the output events array
outEventsEvol_list = [np.zeros((N_sim, networkList[i])).astype(bool) for i 
			in range(1, len(networkList))]

# Create the array of membrane potentials
v_memEvol_list = [np.zeros((N_sim, networkList[i])) for i in range(1, len(networkList))]

# Create the array of weights 
weightsEvol_list = [np.zeros((N_sim, networkList[i], networkList[i-1])) 
		for i in range(1, len(networkList))]


# Simulate the network
accuracy = trainSingleImg(poissonImg, label, labelsArray, networkDictList, dt_tau, v_reset, 
			A_ltp, A_ltd, spikeCountArray, v_memEvol_list, outEventsEvol_list,
			weightsEvol_list)

print("\nSpike Count Array")
print(spikeCountArray)
print("\n")

print("\nClassification result")
print(accuracy)
print("\n")

print("\nFinal labels array")
print(labelsArray)
print("\n")




# Transpose the arrays in order to plot them with respect to time
inEvents_evolution = poissonImg.T


plotTrainNetworkResults(networkList, inEvents_evolution, weightsEvol_list,
			outEventsEvol_list, v_memEvol_list, v_th_list)
