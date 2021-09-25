import sys
import scipy.sparse as sp
import matplotlib.pyplot as plt

development = "../"

if development not in sys.path:
	sys.path.insert(1,development)


from createNetwork import createNetwork
from layers import updateExcLayer, updateInhLayer


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

threshold = np.zeros((N_sim, networkList[1]))

for i in range(N_sim):

	updateExcLayer(network, 1, dt_tauDict["exc"], dt_tauDict["thresh"],
			inputSpikes[i])

	updateInhLayer(network, 1)

	excSpikes[i] = network["excLayer1"]["outSpikes"][0]
	inhSpikes[i] = network["excLayer1"]["inhSpikes"][0]

	excPotential[i] = network["excLayer1"]["v"][0] 

	threshold[i] = network["excLayer1"]["vThresh"]


inputSpikes = inputSpikes.T
excSpikes = excSpikes.T 
inhSpikes = inhSpikes .T
excPotential = excPotential.T
threshold = threshold.T

# First excitatory neuron
fig, axs = plt.subplots(6, 1)
fig.tight_layout()

for i in range(2):
	axs[i].plot(inputSpikes[i])
	axs[i].grid()
	axs[i].set_title("Input spikes " + str(i))

for i in range(2):
	axs[i+2].plot(inhSpikes[i+1])
	axs[i+2].grid()
	axs[i+2].set_title("Inhibitory spikes " + str(i+1))


axs[4].plot(excPotential[0])
axs[4].plot(threshold[0])
axs[4].grid()
axs[4].set_title("Membrane potential")

axs[5].plot(excSpikes[0])
axs[5].grid()
axs[5].set_title("Output spikes")

plt.show()

plt.plot(excPotential[0])
plt.plot(threshold[0])
plt.grid()

plt.show()


# # Second excitatory neuron
# fig, axs = plt.subplots(6, 1)
# 
# for i in range(3):
# 	axs[i].plot(inputSpikes[i])
# 	axs[i].grid()
# 
# axs[3].plot(inhSpikes[0])
# axs[3].grid()
# 
# axs[4].plot(excPotential[1])
# axs[4].plot(threshold[1])
# axs[4].grid()
# 
# axs[5].plot(excSpikes[1])
# axs[5].grid()
# 
# plt.show()



# First inhibitory neuron
#fig, axs = plt.subplots(3, 1)

# axs[0].plot(excSpikes[0])
# axs[0].grid()
# 
# axs[1].plot(inhPotential[0])
# axs[1].plot(inhDict["vReset"]*np.ones(inhPotential[0].shape[0]))
# axs[1].grid()
# 
# axs[2].plot(inhSpikes[0])
# axs[2].grid()
# 
# 
# plt.show()



# # Second inhibitory neuron
# fig, axs = plt.subplots(3, 1)
# 
# axs[0].plot(excSpikes[1])
# axs[0].grid()
# 
# axs[1].plot(inhPotential[1])
# axs[1].grid()
# 
# axs[2].plot(inhSpikes[1])
# axs[2].grid()
# 
# 
# plt.show()

