import sys
import scipy.sparse as sp
import matplotlib.pyplot as plt

development = "../"

if development not in sys.path:
	sys.path.insert(1,development)


from createNetwork import createNetwork
from layers import updateExcLayer, updateInhLayer
from utils import createDir


# Initialize the evaluation parameters
from evaluationParameters import *
from files import *


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

network["exc2exc1"]["weights"][0] *= 2
network["exc2exc1"]["weights"][1] *= 2.5

createDir(figuresDir)


for i in range(N_sim):

	updateExcLayer(network, 1, dt_tauDict["exc"], inputSpikes[i])

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


plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)

# fig = plt.figure(figsize = (22, 22))
# grid_fig = fig.add_gridspec(4, 1)
# ax1, ax2, ax3, ax4 = grid_fig.subplots()
# fig.tight_layout(pad = 0.3)


fig = plt.figure(figsize = (20, 4))

plt.plot(inputSpikes[0].astype(int))
plt.grid()
plt.title("Excitatory spikes", fontsize = 30)

plt.savefig(excSpikesFile, format = 'svg', bbox_inches = 'tight', transparent
		= True)



fig = plt.figure(figsize = (20, 4))

plt.plot(inhSpikes[1])
plt.grid()
plt.title("Inhibitory spikes", fontsize = 30)

plt.savefig(inhSpikesFile, format = 'svg', bbox_inches = 'tight', transparent
		= True)


fig = plt.figure(figsize = (20, 4))

plt.plot(excPotential[0])
plt.plot(threshold[0])
plt.grid()
plt.title("Membrane potential", fontsize = 30)

plt.savefig(voltageFile, format = 'svg', bbox_inches = 'tight', transparent
		= True)


fig = plt.figure(figsize = (20, 4))

plt.plot(excSpikes[0])
plt.grid()
plt.title("Output spikes", fontsize = 30)

plt.savefig(outSpikesFile, format = 'svg', bbox_inches = 'tight', transparent
		= True)



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

