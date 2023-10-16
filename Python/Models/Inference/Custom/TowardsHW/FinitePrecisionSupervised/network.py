import numpy as np

from layers import updateExcLayer
from synapses import stdp 



def run(network, networkList, spikesTrains, dt_tauDict, exp_shift, stdpDict,
		mode, constSums, neuron_bitWidth):


	"""
	Run the network over the duration of the input spikes
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
		threshold .

		5) exp_shift: bit shift for the exponential decay.

		6) stdpDict: dictionary containing the STDP parameters.

		7) mode: string. It can be "train" or "test".

		8) neuron_bitWidth: number of bits on which the neuron works.

	OUTPUT:

		1) spikesCounter: NumPy array containing the total amount of 
		generate spike for each neuron.

		2) spikesMonitor: numpy array. Temporal evolution of the spikes.

		3) membraneMonitor: numpy array. Temporal evolution of the
		membrane potential
	"""

	lastLayerSize = networkList[-1]
	lastLayerIndex = len(networkList) - 1
	trainDuration = spikesTrains.shape[0]

	# Initialize the output spikes counter to 0
	spikesCounter = np.zeros((1, lastLayerSize))
	spikesMonitor = np.zeros(trainDuration).astype(bool)
	membraneMonitor = np.zeros(trainDuration).astype(int)

	for i in range(trainDuration):

		# Train the network over a single step
		updateNetwork(networkList, network, spikesTrains[i], dt_tauDict,
				exp_shift, stdpDict, mode, neuron_bitWidth)

		spikesMonitor[i] = network["excLayer" +
				str(lastLayerIndex)]["outSpikes"][0][0]

		membraneMonitor[i] = network["excLayer" +
				str(lastLayerIndex)]["v"][0][0]


		# Update the output spikes counter
		spikesCounter[0][network["excLayer" +
			str(lastLayerIndex)]["outSpikes"][0]] += 1

	if mode == "train":
		# Normalize the weights
		normalizeWeights(network, networkList, constSums)
	
	return spikesCounter, spikesMonitor, membraneMonitor





def updateNetwork(networkList, network, inputSpikes, dt_tauDict, exp_shift,
		stdpDict, mode, neuron_bitWidth):

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
		threshold .

		5) stdpDict: dictionary containing the STDP parameters.

		6) exp_shift: bit shift for the exponential decay.

		7) stdpDict: dictionary containing the STDP parameters.

		8) mode: string. It can be "train" or "test".

		9) neuron_bitWidth: number of bits on which the neuron works.

	"""	

	layerName = "excLayer1"

	# Update the first excitatory layer
	updateExcLayer(network, 1, exp_shift, inputSpikes, neuron_bitWidth[0])

	for layer in range(2, len(networkList)):

		layerName = "excLayer" + str(layer)
		
		# Update the excitatory layer
		updateExcLayer(network, layer, exp_shift,
			network["excLayer" + str(layer - 1)]["outSpikes"][0],
			neuron_bitWidth[layer-1])


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

