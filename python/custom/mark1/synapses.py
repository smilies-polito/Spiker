import numpy as np

def updateWeights(network, layer, stdpDict, currentStep, inputSpikes):

	'''
	Update the weights through STDP. Update spikes' time instants and masks.

	INPUT:

		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.

		3) stdpDict: dictionary containing the STDP parameters.

		4) currentStep: current value of the training loop index. 

		5) inputSpikes: boolean NumPy array containing the input spikes.

	'''
	
	synapseName = "exc2exc" + str(layer)
	layerName = "excLayer" + str(layer)

	# Update output instants for the active neurons
	updateTime(network, synapseName, "t_out",
		network[layerName]["outSpikes"][0], currentStep - 1)

	# Update output mask for the active neurons
	updateMask(network, synapseName, "mask_out", 
		network[layerName]["outSpikes"][0])

	# Update the weights through STDP
	stdp(network, synapseName, layerName, stdpDict["A_ltp"],
		stdpDict["A_ltd"], stdpDict["ltp_dt_tau"], 
		stdpDict["ltd_dt_tau"], currentStep, inputSpikes)
	
	# Update the input instants of the received spikes
	updateMask(network, synapseName, "mask_in", inputSpikes)

	# Update the input mask with the received spikes
	updateTime(network, synapseName, "t_in", inputSpikes, currentStep)

	


def updateTime(network, synapseName, instants, spikes, currentStep):

	'''
	Update the time instants relative to the arrival of a spike.

	INPUT:

		1) network: dictionary of the network.

		2) synapseName: string. Complete name of the synapse, including
		the index of the layer.

		3) instants: string. Can be "t_in" or "t_out".

		4) spikes: boolean NumPy array containing the spikes. 

		5) currentStep: current value of the training loop index. 

	'''

	# Update only the instants that correspond to a spike
	network[synapseName][instants][0][spikes] = currentStep






def updateMask(network, synapseName, mask, spikes):

	'''
	Update the mask with the new incoming spikes.

	INPUT:

		1) network: dictionary of the network.

		2) synapseName: string. Complete name of the synapse, including
		the index of the layer.

		3) mask: string. Can be "mask_in" or "mask_out".

		4) spikes: boolean NumPy array containing the spikes. 

	'''

	network[synapseName][mask][0] = np.logical_or(network[synapseName]\
						[mask][0], spikes)






def stdp(network, synapseName, layerName, A_ltp, A_ltd, ltp_dt_tau, ltd_dt_tau,
	currentStep, inputSpikes):

	'''
	Increase or decrease the weights through STDP.

	INPUT:

		1) network: dictionary of the network.

		2) synapseName: string. Complete name of the synapse, including
		the index of the layer.

		3) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		4) A_ltp: LTP learning rate. 

		5) A_ltd: LTD learning rate. 

		6) ltp_dt_tau: ratio of the time step and the LTP exponential
		time constant.

		7) ltd_dt_tau: ratio of the time step and the LTD exponential
		time constant.

		8) currentStep: current value of the training loop index. 

		9) inputSpikes: boolean NumPy array containing the input spikes.
	'''


	# Increase the weights of the active neurons' synapses through LTP.
	ltp(network, synapseName, layerName, A_ltp, ltp_dt_tau, currentStep)

	# Decrease the weights of the inactive neurons' synapses through LTD.
	ltd(network, synapseName, layerName, A_ltd, ltp_dt_tau, currentStep, 
		inputSpikes)

	network[synapseName]["weights"][network[synapseName]["weights"] < 0] = 0





def ltp(network, synapseName, layerName, A_ltp, dt_tau, currentStep):

	'''
	Increase the weights of the active neurons' synapses through LTP.

	INPUT:

		1) network: dictionary of the network.

		2) synapseName: string. Complete name of the synapse, including
		the index of the layer.

		3) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		4) A_ltp: LTP learning rate. 

		5) dt_tau: ratio of the time step and the LTP exponential
		time constant.
 
		6) currentStep: current value of the training loop index. 
	
	'''


	# Compute difference between out instant and all the input instants
	Delta_t = (currentStep - 1) - network[synapseName]["t_in"][0]

	# Compute the ltp increment for all the inputs
	ltpIncrease = A_ltp * np.exp(-Delta_t * dt_tau)
	
	# Select only the inputs that have already been active
	ltpIncrease *= network[synapseName]["mask_in"][0]

	# Update the synapses of the active neurons
	network[synapseName]["weights"][network[layerName]["outSpikes"][0]] += \
		ltpIncrease







def ltd(network, synapseName, layerName, A_ltd, dt_tau, currentStep, 
	inputSpikes):

	'''
	Decrease the weights of the inactive neurons' synapses through LTD.

	INPUT:

		1) network: dictionary of the network.

		2) synapseName: string. Complete name of the synapse, including
		the index of the layer.

		3) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		4) A_ltd: LTD learning rate. 

		5) dt_tau: ratio of the time step and the LTD exponential
		time constant.
 
		6) currentStep: current value of the training loop index. 

		7) inputSpikes: boolean NumPy array containing the input spikes.
	'''

	# Compute difference between input instant and all the output instants
	Delta_t = network[synapseName]["t_out"][0] - currentStep 

	# Compute the LTD decrement for all the input synapses
	ltdDecrease = A_ltd * np.exp(Delta_t * dt_tau)

	# Select only the active synapses
	ltdDecrease = np.outer(ltdDecrease, inputSpikes)

	# Select only the inactive neurons
	inactiveNeurons = \
		np.logical_and(np.logical_not(network[layerName]["outSpikes"][0]),
		network[synapseName]["mask_out"][0])

	# Update the weights
	network[synapseName]["weights"][inactiveNeurons] -= \
		ltdDecrease[inactiveNeurons]
		



