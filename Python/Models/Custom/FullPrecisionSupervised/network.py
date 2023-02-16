import numpy as np

from layers import updateExcLayer


import matplotlib.pyplot as plt


def run(network, networkList, spikesTrains, dt_tauDict, stdpDict, mode,
	constSums):


	"""
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

		6) mode: string. It can be "train" or "test".

		7) constSums: numpy array. Normalization factor of the weights
		for each layer.

	OUTPUT:

		spikesCounter: NumPy array containing the total amount of 
		generate spike for each neuron.

	"""

	lastLayerSize = networkList[-1]
	lastLayerIndex = len(networkList) - 1
	trainDuration = spikesTrains.shape[0]

	# Initialize the output spikes counter to 0
	spikesCounter = np.zeros((1, lastLayerSize))

	for i in range(trainDuration):

		# Train the network over a single step
		updateNetwork(networkList, network, spikesTrains[i], dt_tauDict,
			stdpDict, mode)

		
		# print(np.sum(network["exc2exc1"]["weights"][:, spikesTrains[i]],
		#  	axis=1, dtype=np.double)[7])


		# Update the output spikes counter
		spikesCounter[0][network["excLayer" +
			str(lastLayerIndex)]["outSpikes"][0]] += 1


	return spikesCounter





def updateNetwork(networkList, network, inputSpikes, dt_tauDict, stdpDict, 
			mode):

	"""
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

		6) mode: string. It can be "train" or "test".

	"""	

	layerName = "excLayer1"

	# Update the first excitatory layer
	updateExcLayer(network, 1, dt_tauDict["exc"], inputSpikes)


	for layer in range(2, len(networkList)):

		layerName = "excLayer" + str(layer)
		
		# Update the excitatory layer
		updateExcLayer(network, layer, dt_tauDict["exc"],
			network["excLayer" + str(layer - 1)]["outSpikes"][0])

def rest(network, networkList):

	"""
	Bring the network into a rest state.

	INPUT:

		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.
	"""

	for layer in range(1, len(networkList)):

		# Reset the membrane potential to the rest value
		network["excLayer" + str(layer)]["v"][0][:] = network["excLayer"
			+ str(layer)]["vRest"]
