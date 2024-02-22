#!/Users/alessio/anaconda3/bin/python3

import numpy as np
from utils import expDecay


def updateExcLayer(network, layer, dt_tau_exc, inputSpikes):

	"""
	One training step update of the excitatory layer.

	INPUT:
	
		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.

		3) dt_tau_exc: ratio of the time step and the membrane exponential
		time constant.

		4) inputSpikes: boolean NumPy array containing the input spikes.

	"""

	layerName = "excLayer" + str(layer)
	
	# Generate spikes if the potential exceeds the dynamic threshold	
	generateOutputSpikes(network, layerName)

	# Reset the potential of the active neurons	
	resetPotentials(network, layerName)

	# Exponentially decrease the membrane potential
	expDecay(network, layerName, dt_tau_exc, 
		network[layerName]["vRest"], "v")

	if np.sum(inputSpikes) != 0:

		# Update membrane potential with the spikes from the previous layer
		all2allUpdate(network, layerName, "exc2exc" + str(layer), inputSpikes)


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
	network[layerName]["v"][0][network[layerName]["outSpikes"][0]] - \
	network[layerName]["vThresh0"]



def all2allUpdate(network, layerName, synapseName, inputSpikes):

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
