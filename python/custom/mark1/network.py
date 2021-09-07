from layers import updateExcLayer, updateInhLayer
from synapses import updateWeights


def updateNetwork(networkList, network, dt_tau_exc, dt_tau_inh, dt_tau_theta,
			inputSpikes, stdpDict, currentStep):

	# Update the first excitatory layer
	updateExcLayer(network, 1, dt_tau_exc, dt_tau_theta, inputSpikes)

	# Update the first inhibitory layer
	updateInhLayer(network, 1, dt_tau_inh)

	# Update the first layer weights 
	updateWeights(network, 1, stdpDict, currentStep, inputSpikes)


	for layer in range(2, len(networkList)):
		
		# Update the excitatory layer
		updateExcLayer(network, layer, dt_tau_exc, dt_tau_theta,
			network["excLayer" + str(layer - 1)]["outSpikes"][0])
		
		# Update the inhibitory layer
		updateInhLayer(network, layer, dt_tau_inh)

		# Update the layer weights
		updateWeights(network, layer, stdpDict, currentStep,
			network["excLayer" + str(layer - 1)]["outSpikes"][0])


