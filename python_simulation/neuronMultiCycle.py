#!/Users/alessio/anaconda3/bin/python3

# Function which simulates the behaviour of a neuron with multiple
# inputs.
#
# inEvents is expected to be a NumPy array, easy to use in case
# of the need of random inputs.
#
# v_mem, v_th_max and v_th_min are lists containing the
# parameters associated to the single neurons in the previous layer
#
# weights is a NumPy array containing one coefficient for each neuron 
# in the previous layer
#
# dt_tau is a single value, common to all the neurons in the previous
# layer

def neuronMultiCycle(inEvents, v_mem, v_th_max, v_th_min,
			weights, dt_tau):

	previousNode = 0
	outEvent = 0

		
	# Generate event
	if v_mem > v_th_max:
		v_mem = 0
		outEvent = 1



	# Loop for all the events in the previous layer of neurons
	# until the current neuron emits a spike or the previous
	# layer has been completely analized
	while previousNode < inEvents.size:

		# Update with the coefficient corresponding to the
		# neuron in the previous layer
		if inEvents[previousNode] == 1:
			v_mem = v_mem + weights[previousNode]

		# Analize a new node in the previous layer
		previousNode += 1

	
	# Exponential decrease
	if v_mem > v_th_min:
		v_mem = v_mem - dt_tau*v_mem
	else:
		v_mem = 0

	
	return [v_mem, outEvent]
