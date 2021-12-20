#!/Users/alessio/anaconda3/bin/python3

import numpy as np
from utils import expDecay


def updateExcLayer(network, layer, exp_shift, inputSpikes):

	'''
	One training step update of the excitatory layer.

	INPUT:
	
		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.

		3) exp_shift: bit shift for the exponential decay.
		
		4) inputSpikes: boolean NumPy array containing the input spikes.

	'''

	layerName = "excLayer" + str(layer)
	
	# Generate spikes if the potential exceeds the dynamic threshold	
	generateOutputSpikes(network, layerName)

	# Reset the potential of the active neurons	
	resetPotentials(network, layerName)

	# Exponentially decrease the membrane potential
	expDecay(network, layerName, exp_shift, "v")


	if np.sum(inputSpikes) != 0:

		# Update membrane potential with the spikes from the previous layer
		all2allUpdate(network, layerName, "exc2exc" + str(layer), inputSpikes)

	if np.sum(network["excLayer" + str(layer)]["inhSpikes"]) != 0:
		# Update membrane potential with the spikes from the inhibitory layer
		all2othersUpdate(network, layerName, "inh2exc" +
			str(layer))

	# Increase threshold for active neurons. Decrease it for inactive ones.
	homeostasis(network, layerName)





def updateInhLayer(network, layer):

	'''
	One training step update of the inhibitory layer.

	INPUT:
	
		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.

	'''

	layerName = "excLayer" + str(layer)

	network[layerName]["inhSpikes"] = network[layerName]["outSpikes"]






def generateOutputSpikes(network, layerName): 

	''' 
	Generate the output spikes for all those neurons whose membrane
	potential exceeds the threshold (variable with the homeostasis).

	INPUT: 

		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.
	'''
	
	network[layerName]["outSpikes"][0] = network[layerName]["v"][0] > \
		network[layerName]["vThresh"]




def resetPotentials(network, layerName):

	''' 
	Reset the membrane potential of all those neurons whose membrane
	potential has exceeded the threshold. 

	INPUT:
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.

	'''


	network[layerName]["v"][0][network[layerName]["outSpikes"][0]] = \
		network[layerName]["vReset"]




def all2allUpdate(network, layerName, synapseName, inputSpikes):

	''' 
	Update the membrane potential of a fully connected layer.

	INPUT:
		
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		3) synapseName: string. Complete name of the synapse, including the
		index of the synapse itself.

		4) inputSpikes: boolean NumPy array containing the input spikes.

	'''

	network[layerName]["v"][0] = network[layerName]["v"][0] + np.sum(
				network[synapseName]["weights"][:, inputSpikes],
				axis=1)




def all2othersUpdate(network, layerName, synapseName):

	''' 
	Update the membrane potential of a layer with a connection of type
	i!=j, with i = index of a neuron in the origin layer and j = index of a
	neuron in the target layer.

	INPUTS:

		1) network: dictionary of the network.

		2) layerName: strig. Name of the excitatory layer to update.

		3) synapseName: string. Complete name of the synapse, including the
		index of the synapse itself.

	'''

	# Update the potential of the neurons excluded from one spike
	network[layerName]["v"][0][network[layerName]["inhSpikes"][0]] += \
		int(network[synapseName]["weight"] * 
		unconnectedSpikes(network[layerName]["inhSpikes"][0]))

	
	# Update the potential of all the other neurons
	network[layerName]["v"][0]\
		[np.logical_not(network[layerName]["inhSpikes"][0])] +=\
		int(network[synapseName]["weight"] * 
		allSpikes(network[layerName]["inhSpikes"][0]))






def unconnectedSpikes(spikes):

	'''
	Compute the amount of spikes received by a neuron which is excluded by
	one spike.

	INPUT: 
		spikes: boolean NumPy array containing the input spikes.

	OUTPUT:
		total amount of received spikes.
	'''

	# Add together all the received spikes
	totalSpikes = np.sum(spikes)

	if totalSpikes > 0:

		# Remove one spike from the count
		return totalSpikes - 1
	else:
		# Zero spikes	
		return totalSpikes

	



def allSpikes(spikes):

	'''
	Comput the amount of spikes received by a neuron which is connected to
	all the input spikes.

	INPUT:

		spikes: boolean NumPy array containing the input spikes.

	OUTPUT: 
		total amount of received spikes
	'''

	# Add together all the received spikes
	return np.sum(spikes)

	



def homeostasis(network, layerName):

	'''
	Increase the threshold for the active neurons, decrease it for the
	inactive ones.

	INPUT:
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.
	'''

	# Increase homeostasis of the active neurons
	network[layerName]["vThresh"][0][network[layerName]["outSpikes"][0]] += \
		network[layerName]["vThreshPlus"]
