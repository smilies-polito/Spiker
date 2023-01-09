import sys
import scipy.sparse as sp
import matplotlib.pyplot as plt

development = "../"

if development not in sys.path:
	sys.path.insert(1,development)


from createNetwork import createNetwork
from network import updateNetwork


# Initialize the evaluation parameters
from evaluationParameters import *


# Create the network
network = createNetwork(networkList, None, None, mode, excDictList, 
			scaleFactors, inh2excWeights)


# Create a spare matrix of random spikes
inputSpikes = sp.random(N_sim, networkList[0], density = density)
inputSpikes = inputSpikes.A.astype(bool)

# Preallocate the temporal evolution arrays
excSpikes = np.zeros((N_sim, networkList[1])).astype(bool)
inhSpikes = np.zeros((N_sim, networkList[1])).astype(bool)

excPotential = np.zeros((N_sim, networkList[1]))
inhPotential = np.zeros((N_sim, networkList[1]))

threshold = np.zeros((N_sim, networkList[1]))

for i in range(N_sim):

	updateNetwork(networkList, network, inputSpikes[i], dt_tauDict, 
		stdpDict, mode)

	inhSpikes[i] = network["excLayer1"]["inhSpikes"][0]

	excPotential[i] = network["excLayer1"]["v"][0]

	threshold[i] = network["excLayer1"]["vThresh"]


inputSpikes = inputSpikes.T
inhSpikes = inhSpikes .T
excPotential = excPotential.T
threshold = threshold.T

# First excitatory neuron
fig, axs = plt.subplots(5, 1)

for i in range(3):
	axs[i].plot(inputSpikes[i])
	axs[i].grid()

axs[3].plot(inhSpikes[1])
axs[3].grid()

axs[4].plot(excPotential[0])
axs[4].plot(threshold[0])
axs[4].grid()

plt.show()


# Second excitatory neuron
fig, axs = plt.subplots(5, 1)

for i in range(3):
	axs[i].plot(inputSpikes[i])
	axs[i].grid()

axs[3].plot(inhSpikes[0])
axs[3].grid()

axs[4].plot(excPotential[1])
axs[4].plot(threshold[1])
axs[4].grid()

plt.show()
