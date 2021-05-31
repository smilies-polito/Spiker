#!/Users/alessio/anaconda3/bin/python3


# Function which simulates the behaviour of a simple spiking neuron with
# multiple inputs. This is a first model of a neuron that can be inserted in a
# network and used for practical computations.
#
# INPUT PARAMETERS:
#
# 	1) inEvents: binary NumPy array that represents the presence of an event
# 	on each input.
#
# 	2) v_mem: membrane potential of the neuron. It is made evolve using the
# 	developed model of decresing exponential + increment in case of the
# 	presence of one or more events.
#
# 	3) v_th_max and v_th_min are two hyperparameters that need to be
# 	properly tuned during the learning phase. In particular v_th_min still
# 	needs to be verified in terms of the maximum error introduced with
# 	respect to a pure exponential decrease.
#
# 	4) weight is a NumPy array which represents the importance given to each
# 	input synapse in making the neuron fire. This is another hyperparameter
# 	that needs to be modified during the learning phase.
#
# 	5) dt_tau is the last hyperparameter that can be modified and represents
# 	the ratio between the considered time step and the exponential time
# 	constant

def neuronMultiCycle(inEvents, v_mem, v_th_max, v_th_min, weights, dt_tau,
	prevLayerDim):

	outEvent = 0

	# Generate event
	if v_mem > v_th_max: 
		v_mem = 0 
		outEvent = 1


	# Loop for all the events in the previous layer of neurons until the
	# current neuron emits a spike or the previous layer has been completely
	# analized
	for previousNode in range(prevLayerDim):

		# Update with the coefficient corresponding to the neuron in the
		# previous layer
		if inEvents[previousNode] == 1: v_mem = v_mem +\
			weights[previousNode]

		# Analize a new node in the previous layer
		previousNode += 1

	
	# Exponential decrease
	v_mem = v_mem - dt_tau*v_mem

	return [v_mem, outEvent]
