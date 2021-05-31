#!/Users/alessio/anaconda3/bin/python3

# Function that models the behaviour of a fully connected spiking neural
# network, composed by the desired amount of layers, which in turns contain the
# desired amount of multi cycle neurons.
#
# ARGUMENTS: 
#
# 	1) inEvents is a NumPy array that contains the events provided as
# 	an input of the net.
#
# 	2) layersList is a list containing the desctiption of the network. It
# 	has a number of entries equal to the total number of layers in the net.
# 	Each entry represents the number of neurons in the relative layer. 
# 
# Being each layer's dimension potentially different from all the others the
# most convenient data structure to model the network is the list. In particular
# each entry of the list is associated to a layer and consists of a data
# structure that stores the parametes of each neuron in the layer. Since each
# layer is updated using the function "layer" each list, associated to one
# parameter of the net, contains the proper data structure expected by the
# function "layer", generally a NumPy array:
#
# 	3) v_mem, v_th_max, v_th_min are lists of NumPy arrays. Each neuron has
# 	its own membrane potential, firing threshold and minimum threshold and
# 	so each entry of the three lists is associated with a layer and each
# 	entry of the NumPy arrays is associated to a neuron.
#
# 	4) weights is a list of bidimensional NumPy arrays containing, for each
# 	neuron in each layer, a number of weights that corresponds to the number
# 	of nodes in the previous layer, one for each input synapse.
#
# 	5) dt_tau is a numerical value. It is an intrinsic parameter of the
# 	exponential model and for this reason it is the same for all the
# 	neurons.
#
# 	6) outEvents is a list of NumPy arrays. Each of these arrays is filled
# 	by the function with the values (0 or 1) of the events generated by each
# 	neuron in the network.
#
# Being the NumPy arrays a mutable variable type the function doesn't return
# anything but simply updates the values contained inside the arrays that are
# passed as an input argument.  This is useful from a performance point of view:
# the whole amount of arrays needed for the evolution of the net can be
# preallocated at the beginning of the simulation and then simply updated by the
# various functions.

from layerMultiCycle import layerMultiCycle

def snnMultiCycle(inEvents, layersList, v_mem, v_th_max, v_th_min, weights,
	dt_tau, outEvents):


	# Update the first layer with the input events
	layerMultiCycle(inEvents, v_mem[0], v_th_max[0], v_th_min[0],
		weights[0], dt_tau, layersList[0], inEvents.size, outEvents[0])

	# Loop over all the other layers in the network updating them using the
	# output of the previous layer as an input
	for i in range(1, len(layersList)):

		# Update the layer
		layerMultiCycle(outEvents[i-1], v_mem[i], v_th_max[i], 
			v_th_min[i], weights[i], dt_tau, layersList[i],
			layersList[i-1], outEvents[i])
