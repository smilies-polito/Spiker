import numpy as np
from utils import expDecay


def stdp(network, layer, stdpDict, inputSpikes):

	'''
	Increase or decrease the weights through STDP.

	INPUT:

		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.
 
		4) stdpDict: dictionary containing the STDP parameters.

		5) inputSpikes: boolean NumPy array containing the input spikes.
	'''

	synapseName = "exc2exc" + str(layer)
	layerName = "excLayer" + str(layer)

	# Increase the weights of the active neurons' synapses through LTP.
	ltp(network, synapseName, layerName, stdpDict["eta_post"],
		stdpDict["ltp_dt_tau"])

	# Decrease the weights of the inactive neurons' synapses through LTD.
	ltd(network, synapseName, layerName, stdpDict["eta_pre"],
		stdpDict["ltd_dt_tau"], inputSpikes)

	
	network[synapseName]["weights"][network[synapseName]["weights"] < 0] = 0





def ltp(network, synapseName, layerName, eta_pre, dt_tau):

	'''
	Increase the weights of the active neurons' synapses through LTP.

	INPUT:

		1) network: dictionary of the network.

		2) synapseName: string. Complete name of the synapse, including
		the index of the layer.

		3) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		4) A_ltd: LTD learning rate. 

		5) dt_tau: ratio of the time step and the LTP exponential
		time constant.
 
	'''

	# Exponentially decrease the post-synaptic trace
	expDecay(network, synapseName, dt_tau, 0, "post")

	# Reset the post-synaptic trace to its starting value
	network[synapseName]["post"][:, 0][network[layerName]["outSpikes"][0]] = \
		eta_pre

	# Update the synapses of the active neurons
	network[synapseName]["weights"][network[layerName]["outSpikes"][0]] += \
		network[synapseName]["pre"]





def ltd(network, synapseName, layerName, eta_post, dt_tau, inputSpikes):

	'''
	Decrease the weights of the inactive neurons' synapses through LTD.

	INPUT:

		1) network: dictionary of the network.

		2) synapseName: string. Complete name of the synapse, including
		the index of the layer.

		3) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		4) A_ltp: LTP learning rate. 

		5) dt_tau: ratio of the time step and the LTD exponential
		time constant.
 
		6) inputSpikes: boolean NumPy array containing the input spikes.
	'''
	
	# Exponentially decrease the pre-synaptic trace
	expDecay(network, synapseName, dt_tau, 0, "pre")
	
	# Reset the pre-synaptic trace to its starting value
	network[synapseName]["pre"][0][inputSpikes] = eta_post

	# Update the synapses of the inactive neurons
	network[synapseName]["weights"][:, inputSpikes] -= \
		network[synapseName]["post"]
