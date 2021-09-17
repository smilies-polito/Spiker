import numpy as np

from layers import updateExcLayer, updateInhLayer
from synapses import stdp 


import matplotlib.pyplot as plt

def run(network, networkList, spikesTrains, dt_tauDict, stdpDict):


	'''
	Run the training of the network over the duration of the input spikes
	trains.

	INPUT:
		
		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) spikesTrains: two-dimensional NumPy array. One training step
		for each row; one input for each column.

		4) dt_tauDict: dictionary containing the exponential constants
		of the excitatory and inhibitory membrane and of the 
		homeostasis parameter theta .

		5) stdpDict: dictionary containing the STDP parameters.

	OUTPUT:

		spikesCounter: NumPy array containing the total amount of 
		generate spike for each neuron.

	'''

	lastLayerSize = networkList[-1]
	lastLayerIndex = len(networkList) - 1
	trainDuration = spikesTrains.shape[0]

	# Initialize the output spikes counter to 0
	spikesCounter = np.zeros((1, lastLayerSize))

	for i in range(trainDuration):

		# Train the network over a single step
		updateNetwork(networkList, network, spikesTrains[i], dt_tauDict,
			stdpDict)

		# Update the output spikes counter
		spikesCounter[0][network["excLayer" +
			str(lastLayerIndex)]["outSpikes"][0]] += 1


	return spikesCounter





def updateNetwork(networkList, network, inputSpikes, dt_tauDict, stdpDict):

	'''
	One training step update for the entire network.

	INPUT:

		1) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) network: dictionary of the network.

		3) inputSpikes: NumPy array. Value of the input spikes within a
		single training step, one for each neuron.
		
		4) dt_tauDict: dictionary containing the exponential constants
		of the excitatory and inhibitory membrane and of the 
		homeostasis parameter theta .

		5) stdpDict: dictionary containing the STDP parameters.

		6) currentStep: current value of the training loop index. 

	'''	

	# Update the first excitatory layer
	updateExcLayer(network, 1, dt_tauDict["exc"], dt_tauDict["theta"],
		inputSpikes)

	# Update the first inhibitory layer
	updateInhLayer(network, 1, dt_tauDict["inh"])

	# Update the first layer weights 
	stdp(network, 1, stdpDict, inputSpikes)


	for layer in range(2, len(networkList)):
		
		# Update the excitatory layer
		updateExcLayer(network, layer, dt_tauDict["exc"],
			dt_tauDict["theta"], network["excLayer" + 
			str(layer - 1)]["outSpikes"][0])
		
		# Update the inhibitory layer
		updateInhLayer(network, layer, dt_tauDict["inh"])

		# Update the layer weights
		stdp(network, layer, stdpDict, network["excLayer" + 
			str(layer - 1)]["outSpikes"][0])


