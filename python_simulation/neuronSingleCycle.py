#!/Users/alessio/anaconda3/bin/python3

# Function which simulates the behaviour of a simple spiking
# neuron with a single input (it can process one event per
# cycle)
def neuronSingleCycle(inEvent, v_mem, v_th_max, v_th_min, weight, dt_tau):

	# Check if the previous update has caused the 
	# membrane potential to grow over the threshold
	if v_mem > v_th_max:

		# Fire a spike
		outEvent = 1

		# Reset the membrane potential
		v_mem = 0

	else:
		outEvent = 0

		# If the neuron receives a new event in input
		if inEvent == 1:

			# Update the membrane potential adding
			# the weight of the synapse that
			# corresponds to the neuron that has
			# generated the event
			v_mem = v_mem - dt_tau*v_mem + weight
		else:
			# If no event happened: exponential
			# decay of the membrane potential
			v_mem = v_mem - dt_tau*v_mem

			# If the decay has reduced the potential
			# below a choosen threshold reset it
			if v_mem < v_th_min:
				v_mem = 0

	return [v_mem, outEvent]
