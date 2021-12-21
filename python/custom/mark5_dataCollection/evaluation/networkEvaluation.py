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
			inhDictList, scaleFactors, exc2inhWeights,
			inh2excWeights)


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

	excSpikes[i] = network["excLayer1"]["outSpikes"][0]
	inhSpikes[i] = network["inhLayer1"]["outSpikes"][0]

	excPotential[i] = network["excLayer1"]["v"][0]
	inhPotential[i] = network["inhLayer1"]["v"][0]

	threshold[i] = network["excLayer1"]["vThresh"] + \
	network["excLayer1"]["theta"]


inputSpikes = inputSpikes.T
excSpikes = excSpikes.T 
inhSpikes = inhSpikes .T
excPotential = excPotential.T
inhPotential = inhPotential.T
threshold = threshold.T

# First excitatory neuron
fig, axs = plt.subplots(6, 1)

for i in range(3):
	axs[i].plot(inputSpikes[i])
	axs[i].grid()

axs[3].plot(inhSpikes[1])
axs[3].grid()

axs[4].plot(excPotential[0])
axs[4].plot(threshold[0])
axs[4].grid()

axs[5].plot(excSpikes[0])
axs[5].grid()

plt.show()


# Second excitatory neuron
fig, axs = plt.subplots(6, 1)

for i in range(3):
	axs[i].plot(inputSpikes[i])
	axs[i].grid()

axs[3].plot(inhSpikes[0])
axs[3].grid()

axs[4].plot(excPotential[1])
axs[4].plot(threshold[1])
axs[4].grid()

axs[5].plot(excSpikes[1])
axs[5].grid()

plt.show()



# First inhibitory neuron
fig, axs = plt.subplots(3, 1)

axs[0].plot(excSpikes[0])
axs[0].grid()

axs[1].plot(inhPotential[0])
axs[1].grid()

axs[2].plot(inhSpikes[0])
axs[2].grid()


plt.show()



# Second inhibitory neuron
fig, axs = plt.subplots(3, 1)

axs[0].plot(excSpikes[1])
axs[0].grid()

axs[1].plot(inhPotential[1])
axs[1].grid()

axs[2].plot(inhSpikes[1])
axs[2].grid()


plt.show()

