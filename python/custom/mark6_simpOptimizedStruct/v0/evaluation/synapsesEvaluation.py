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

np.set_printoptions(linewidth=150)


# Create the network
network = createNetwork(networkList, None, None, mode, excDictList, 
			inhDictList, scaleFactors, exc2inhWeights,
			inh2excWeights)

# Create a spare matrix of random spikes
inputSpikes = sp.random(N_sim, networkList[0], density = density)
inputSpikes = inputSpikes.A.astype(bool)


# Preallocate the temporal evolution arrays
excSpikes = np.zeros((N_sim, networkList[1])).astype(bool)
weights = np.zeros((N_sim, networkList[1], networkList[0]))
post = np.zeros((N_sim, networkList[1]))
pre = np.zeros((N_sim, networkList[0]))


for i in range(N_sim):

	# Update excitatory layer
	updateExcLayer(network, 1, dt_tauDict["exc"], dt_tauDict["theta"], inputSpikes[i])
	
	# Update inhibitory layer
	updateInhLayer(network, 1, dt_tauDict["inh"])

	# Update the synapse's weights
	stdp(network, 1, stdpDict, inputSpikes[i])

	post[i]		= network["exc2exc1"]["post"][:, 0]
	pre[i]		= network["exc2exc1"]["pre"][0]
	weights[i]	= network["exc2exc1"]["weights"]
	excSpikes[i]	= network["excLayer1"]["outSpikes"][0]


inputSpikes = inputSpikes.T
post = post.T
pre = pre.T
weights = np.transpose(weights, (1,2,0))
excSpikes = excSpikes.T




# First excitatory neuron
# ------------------------------------------------------------------------------

# First input
fig, axs = plt.subplots(2, 1)

axs[0].plot(inputSpikes[0])
axs[0].grid()
axs[0].set_title("Input spikes")

axs[1].plot(pre[0])
axs[1].grid()
axs[1].set_title("Pre-synaptic trace")

plt.show()

#axs[2].plot(weights[0][0])
#axs[2].grid()

fig, axs = plt.subplots(2, 1)


axs[0].plot(excSpikes[0])
axs[0].set_title("Output spikes")
axs[0].grid()

axs[1].plot(post[0])
axs[1].grid()
axs[1].set_title("Post-synaptic trace")

plt.show()

# 
# 
# # Second input
# fig, axs = plt.subplots(5, 1)
# 
# axs[0].plot(inputSpikes[1])
# axs[0].grid()
# 
# axs[1].plot(pre[1])
# axs[1].grid()
# 
# axs[2].plot(weights[0][1])
# axs[2].grid()
# 
# axs[3].plot(excSpikes[0])
# axs[3].grid()
# 
# axs[4].plot(post[0])
# axs[4].grid()
# 
# 
# plt.show()
# 
# 
# 
# # Third input
# fig, axs = plt.subplots(5, 1)
# 
# axs[0].plot(inputSpikes[2])
# axs[0].grid()
# 
# axs[1].plot(pre[2])
# axs[1].grid()
# 
# axs[2].plot(weights[0][2])
# axs[2].grid()
# 
# axs[3].plot(excSpikes[0])
# axs[3].grid()
# 
# axs[4].plot(post[0])
# axs[4].grid()
# 
# 
# plt.show()
# 
# 
# 
# # Second excitatory neuron
# # ------------------------------------------------------------------------------
# 
# # First input
# fig, axs = plt.subplots(5, 1)
# 
# axs[0].plot(inputSpikes[0])
# axs[0].grid()
# 
# axs[1].plot(pre[0])
# axs[1].grid()
# 
# axs[2].plot(weights[1][0])
# axs[2].grid()
# 
# axs[3].plot(excSpikes[1])
# axs[3].grid()
# 
# axs[4].plot(post[1])
# axs[4].grid()
# 
# 
# plt.show()
# 
# 
# # Second input
# fig, axs = plt.subplots(5, 1)
# 
# axs[0].plot(inputSpikes[1])
# axs[0].grid()
# 
# axs[1].plot(pre[1])
# axs[1].grid()
# 
# axs[2].plot(weights[1][1])
# axs[2].grid()
# 
# axs[3].plot(excSpikes[1])
# axs[3].grid()
# 
# axs[4].plot(post[1])
# axs[4].grid()
# 
# 
# plt.show()
# 
# 
# 
# # Third input
# fig, axs = plt.subplots(5, 1)
# 
# axs[0].plot(inputSpikes[2])
# axs[0].grid()
# 
# axs[1].plot(pre[2])
# axs[1].grid()
# 
# axs[2].plot(weights[1][2])
# axs[2].grid()
# 
# axs[3].plot(excSpikes[1])
# axs[3].grid()
# 
# axs[4].plot(post[1])
# axs[4].grid()
# 
# 
# plt.show()
