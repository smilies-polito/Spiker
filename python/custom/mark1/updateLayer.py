#!/Users/alessio/anaconda3/bin/python3

import numpy as np



def updateExcLayer(network, layer, dt_tau, inputSpikes):

	'''
	One training step update of the excitatory layer.

	INPUT:
	
		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.

		3) dt_tau: ratio of the time step and the membrane exponential
		time constant.

		4) inputSpikes: boolean NumPy array containing the input spikes.

	'''

	originLayer = "inhLayer" + str(layer)
	targetLayer = "excLayer" + str(layer)
	
	# Generate spikes if the potential exceeds the dynamic threshold	
	generateOutputSpikes(network, targetLayer)

	# Reset the potential of the active neurons	
	resetPotentials(network, targetLayer)

	# Exponentially decrease the membrane potential
	expDecay(network, targetLayer, dt_tau, 
		network[targetLayer]["vRest"], "v")

	# Update membrane potential with the spikes from the previous layer
	all2allUpdate(network, targetLayer, "exc2exc" + str(layer), inputSpikes)

	# Update membrane potential with the spikes from the inhibitory layer
	all2othersUpdate(network, originLayer, targetLayer, "inh2exc" +
		str(layer))

	# Increase threshold for active neurons. Decrease it for inactive ones.
	homeostasis(network, layerName, dt_tau)





def updateInhLayer(network, layer, dt_tau):

	'''
	One training step update of the inhibitory layer.

	INPUT:
	
		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.

		3) dt_tau: ratio of the time step and the membrane exponential
		time constant.
	'''

	originLayer = "excLayer" + str(layer)
	targetLayer = "inhLayer" + str(layer)

	
	# Generate spikes if the potential exceeds the dynamic threshold	
	generateOutputSpikes(network, targetLayer)
	
	# Reset the potential of the active neurons	
	resetPotentials(network, targetLayer)
	
	# Exponentially decrease the membrane potential
	expDecay(network, targetLayer, dt_tau, 
		network[targetLayer]["vRest"], "v")

	
	# Update membrane potential with the spikes from the excitatory layer
	one2oneUpdate(network, originLayer, targetLayer, "exc2inh" + str(layer))





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
		network[layerName]["vThresh"] + network[layerName]["theta"]





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





def expDecay(network, layerName, dt_tau, steadyStateValue, stateVariable): 
	
	''' 
	Decrease the desired state variable belonging to the layer dictionary
	with exponential decay.

	INPUT: 

		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.
		
		3) dt_tau: ratio of the time step and the exponential time
		constant.

		4) steadyStateValue: value towards which the decreasing
		exponential tends.

		5) stateVariable: string. Name of the variable to update.
	'''

	network[layerName][stateVariable][0] = \
		network[layerName][stateVariable][0] - \
		dt_tau * (network[layerName][stateVariable][0] -
		steadyStateValue)







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




def all2othersUpdate(network, originLayer, targetLayer, synapseName):

	''' 
	Update the membrane potential of a layer with a connection of type
	i!=j, with i = index of a neuron in the origin layer and j = index of a
	neuron in the target layer.

	INPUTS:

		1) network: dictionary of the network.

		2) originLayer: string. Layer inside the network that has
		generated the input spikes.

		3) targetLayer: string. Layer inside the network that reveives
		the input spikes.

		4) synapseName: string. Complete name of the synapse, including the
		index of the synapse itself.

	'''

	# Update the potential of the neurons excluded from one spike
	network[targetLayer]["v"][0][network[originLayer]["outSpikes"][0]] += \
		network[synapseName]["weight"] * \
		unconnectedSpikes(network[originLayer]["outSpikes"][0])

	
	# Update the potential of all the other neurons
	network[targetLayer]["v"][0]\
		[np.logical_not(network[originLayer]["outSpikes"][0])] +=\
		network[synapseName]["weight"] * \
		allSpikes(network[originLayer]["outSpikes"][0])






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

	


def one2oneUpdate(network, originLayer, targetLayer, synapseName):

	''' 
	Update the membrane potential of a layer with a connection of type
	i=j, with i = index of a neuron in the origin layer and j = index of a
	neuron in the target layer.

	INPUTS:

		1) network: dictionary of the network.

		2) originLayer: string. Layer inside the network that has
		generated the input spikes.

		3) targetLayer: string. Layer inside the network that reveives
		the input spikes.

		4) synapseName: string. Complete name of the synapse, including the
		index of the synapse itself.
		
	'''

	network[targetLayer]["v"][0][network[originLayer]["outSpikes"][0]] += \
		network[synapseName]["weight"]




def homeostasis(network, layerName, dt_tau):

	'''
	Increase the threshold for the active neurons, decrease it for the
	inactive ones.

	INPUT:
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		3) dt_tau: ratio of the time step and the exponential time
		constant.
	'''

	# Exponentially decrease the dynamic homeostasis
	expDecay(network, layerName, dt_tau, 0, "theta")

	# Increase homeostasis of the active neurons
	increaseHomeostasis(network, layerName)




def increaseHomeostasis(network, layerName):

	''' 
	Increase the homeostasis of all those neurons whose membrane
	potential has exceeded the threshold. 

	INPUT:
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.
	'''

	network[layerName]["theta"][0][network[layerName]["outSpikes"][0]] += \
		network[layerName]["thetaPlus"]
