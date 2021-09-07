import sys
import scipy.sparse as sp
import matplotlib.pyplot as plt

development = "../"

if development not in sys.path:
	sys.path.insert(1,development)


from createNetwork import createNetwork
from layers import updateExcLayer, updateInhLayer
from synapses import *


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
weights = np.zeros((N_sim, networkList[1], networkList[0]))



for i in range(N_sim):

	# Update excitatory layer
	updateExcLayer(network, 1, dt_tau_exc, dt_tau_theta, inputSpikes[i])
	
	# Update inhibitory layer
	updateInhLayer(network, 1, dt_tau_inh)

	# Update the synapse's weights
	updateWeights(network, "exc2exc1", "excLayer1", stdpDict, i,
	inputSpikes[i])

	excSpikes[i] = network["excLayer1"]["outSpikes"][0]
	weights[i] = network["exc2exc1"]["weights"]



inputSpikes = inputSpikes.T
excSpikes = excSpikes.T 
weights = np.transpose(weights, (1,2,0))




# First excitatory neuron
# ------------------------------------------------------------------------------

# First input
fig, axs = plt.subplots(3, 1)

axs[0].plot(inputSpikes[0])
axs[0].grid()

axs[1].plot(weights[0][0])
axs[1].grid()

axs[2].plot(excSpikes[0])
axs[2].grid()

plt.show()


# Second input
fig, axs = plt.subplots(3, 1)

axs[0].plot(inputSpikes[1])
axs[0].grid()

axs[1].plot(weights[0][1])
axs[1].grid()

axs[2].plot(excSpikes[0])
axs[2].grid()

plt.show()



# Third input
fig, axs = plt.subplots(3, 1)

axs[0].plot(inputSpikes[2])
axs[0].grid()

axs[1].plot(weights[0][2])
axs[1].grid()

axs[2].plot(excSpikes[0])
axs[2].grid()

plt.show()



# Second excitatory neuron
# ------------------------------------------------------------------------------

# First input
fig, axs = plt.subplots(3, 1)

axs[0].plot(inputSpikes[0])
axs[0].grid()

axs[1].plot(weights[1][0])
axs[1].grid()

axs[2].plot(excSpikes[1])
axs[2].grid()

plt.show()


# Second input
fig, axs = plt.subplots(3, 1)

axs[0].plot(inputSpikes[1])
axs[0].grid()

axs[1].plot(weights[1][1])
axs[1].grid()

axs[2].plot(excSpikes[1])
axs[2].grid()

plt.show()



# Third input
fig, axs = plt.subplots(3, 1)

axs[0].plot(inputSpikes[2])
axs[0].grid()

axs[1].plot(weights[1][2])
axs[1].grid()

axs[2].plot(excSpikes[1])
axs[2].grid()

plt.show()

