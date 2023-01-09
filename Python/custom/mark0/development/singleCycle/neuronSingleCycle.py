#!/Users/alessio/anaconda3/bin/python3

# Function which simulates the behaviour of a simple spiking
# neuron with a single input. This is the starting point to
# develop a model of a spiking neuron which reacts to
# spiking inputs.
#
# INPUT PARAMETERS:
#
# 	1) inEvent: single binary value, 0 or 1, that
# 	   represents the presence of an event on the
# 	   input.
#
# 	2) v_mem: membrane potential of the neuron. It is
# 	   made evolve using the developed model of
# 	   decresing exponential + increment in case of the
# 	   presence of an event.
#
# 	3) v_th_max and v_th_min are two hyperparameters
# 	   that need to be properly tuned during the learning
# 	   phase. In particular v_th_min still needs to be
# 	   verified in terms of the maximum error introduced
# 	   with respect to a pure exponential decrease.
#
# 	4) weight is a single value which represents the
# 	   importance given to the input synapse in making
# 	   the neuron fire. This is another hyperparameter
# 	   that needs to be modified during the learning
# 	   phase.
#
# 	5) dt_tau is the last hyperparameter that can be
# 	   modified and represents the ratio between the
# 	   considered time step and the exponential time
# 	   constant

def neuronSingleCycle(inEvent, v_mem, v_th_max, v_th_min, weight, 
			dt_tau):

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
