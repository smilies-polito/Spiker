import numpy as np

from synapses import stdp 

from utils import expDecay, checkBitWidth, saturateArray
from bitWidths import fixed_point_decimals, neuron_bitWidth


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
	spikesMonitor_0 = np.zeros((trainDuration, networkList[1])).astype(bool)
	spikesMonitor_1 = np.zeros((trainDuration, lastLayerSize)).astype(bool)
	membraneMonitor = np.zeros((trainDuration, lastLayerSize)).astype(int)

	for i in range(trainDuration):

		# Train the network over a single step
		updateNetwork(networkList, network, spikesTrains[i], dt_tauDict,
				exp_shift, stdpDict, mode, neuron_bitWidth)

		spikesMonitor_0[i] = network["excLayer" +
				str(1)]["outSpikes"][0]

		spikesMonitor_1[i] = network["excLayer" +
				str(lastLayerIndex)]["outSpikes"][0]

		membraneMonitor[i] = network["excLayer" +
				str(lastLayerIndex)]["v"][0]


		# Update the output spikes counter
		spikesCounter[0][network["excLayer" +
			str(lastLayerIndex)]["outSpikes"][0]] += 1

	if mode == "train":
		# Normalize the weights
		normalizeWeights(network, networkList, constSums)
	
	return spikesCounter, spikesMonitor_0, spikesMonitor_1, membraneMonitor





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



def updateExcLayer(network, layer, exp_shift, inputSpikes, neuron_bitWidth):

	"""
	One training step update of the excitatory layer.

	INPUT:
	
		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.

		3) exp_shift: bit shift for the exponential decay.
		
		4) inputSpikes: boolean NumPy array containing the input spikes.

		5) neuron_bitWidth: number of bits on which the neuron works.

	"""

	layerName = "excLayer" + str(layer)
	
	# Generate spikes if the potential exceeds the dynamic threshold	
	generateOutputSpikes(network, layerName)

	# Reset the potential of the active neurons	
	resetPotentials(network, layerName)

	# Exponentially decrease the membrane potential
	expDecay(network, layerName, exp_shift, "v")


	if np.sum(inputSpikes) != 0:

		# Update membrane potential with the spikes from the previous layer
		all2allUpdate(network, layerName, "exc2exc" + str(layer),
				inputSpikes, neuron_bitWidth)


	checkBitWidth(network["excLayer" + str(layer)]["v"],
			neuron_bitWidth)


def generateOutputSpikes(network, layerName): 

	""" 
	Generate the output spikes for all those neurons whose membrane
	potential exceeds the threshold (variable with the homeostasis).

	INPUT: 

		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.
	"""
	
	network[layerName]["outSpikes"][0] = network[layerName]["v"][0] > \
		network[layerName]["vThresh"]




def resetPotentials(network, layerName):

	""" 
	Reset the membrane potential of all those neurons whose membrane
	potential has exceeded the threshold. 

	INPUT:
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.

	"""


	network[layerName]["v"][0][network[layerName]["outSpikes"][0]] = \
		network[layerName]["vReset"]




def all2allUpdate(network, layerName, synapseName, inputSpikes, bitwidth):

	""" 
	Update the membrane potential of a fully connected layer.

	INPUT:
		
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		3) synapseName: string. Complete name of the synapse, including the
		index of the synapse itself.

		4) inputSpikes: boolean NumPy array containing the input spikes.

	"""

	network[layerName]["v"][0] = network[layerName]["v"][0] + np.sum(
				network[synapseName]["weights"][:, inputSpikes],
				axis=1)

	network[layerName]["v"][0] = saturateArray(network[layerName]["v"][0],
			bitwidth)

