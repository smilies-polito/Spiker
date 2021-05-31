#!/Users/alessio/anaconda3/bin/python3

# Function that updates an entire layer of neurons.
#
# ARGUMENTS: 
#
# 	1) inEvents is a NumPy array that contains the output of all the
# 	neurons in the previous layer. The output of each of these neurons is
# 	connected to all the nodes in the current layer (fully connected net)
#
# 	2) v_mem, v_th_max, v_th_min are NumPy arrays that contain the
# 	parameters of the neurons of the current layer. Each neuron has its own
# 	membrane potential, firing threshold and minimum threshold
#
# 	3) weights is a bidimensional NumPy array containing, for each neuron in
# 	the current layer, a number of weights that corresponds to the number of
# 	nodes in the previous layer, one for each input synapse.
#
# 	4) dt_tau is a numerical value. It is an intrinsic parameter of the
# 	exponential model and for this reason it is the same for all the neurons
#
# 	5) N_curr and N_prev are respectively the dimensions of the current and
# 	previous layers of neurons, useful to dimension the loops inside the
# 	nested functions
#
# 	6) outEvents is a NumPy array that is filled by the function with the
# 	values (0 or 1) of the events generated by each neuron in the considered
# 	layer
#
# Being the NumPy arrays a mutable variable type the function doesn't return
# anything but simply updates the values contained inside the arrays that ar
# passed as an input argument.  This is useful from a performance point of view:
# the whole amount of arrays needed for the evolution of the net can be
# preallocated at the beginning of the simulation and then simply updated by the
# various functions.

from neuronMultiCycle import neuronMultiCycle


def layerMultiCycle(inEvents, v_mem, v_th_max, v_th_min, weights, dt_tau,
	N_curr, N_prev, outEvents):


	# Loop over all the neurons in the current layer
	for i in range(N_curr):

		# Update the neuron
		update = neuronMultiCycle(inEvents, v_mem[i], v_th_max[i],
			v_th_min[i], weights[i], dt_tau, N_prev)

		# Update the output arrays with the values returned by the
		# neuron
		v_mem[i] = update[0] 
		outEvents[i] = update[1]
