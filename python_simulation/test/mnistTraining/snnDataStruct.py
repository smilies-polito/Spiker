#!/Users/alessio/anaconda3/bin/python3

import numpy as np


# Create a list of dictionaries containing the parameters of all the neurons in the
# network. 
#
#	INPUT PARAMETERS:
#
#		1) v_th_list: list of NumPy arrays containing the threshold values. Each
#		NumPy array is associated to a layer of neurons. In the simplest case all
#		the neurons have the same threshold.
#
#		2) v_reset: numerical value corresponding to the voltage at which the
#		membrane potential is reset. This is common to all the neurons.
#
#		3) w_min_list: list of NumPy arrays. Each array corresponds to a layer.
#		Each element of the arrays corresponds to the minimum value of the range
#		in which the random weights associated to the inputs of a specific neuron
#		are initialized. In the simplest case this value is common to all the
#		neurons.
#
#		4) w_max_list: list of NumPy arrays. Each array corresponds to a layer.
#		Each element of the arrays corresponds to the maximum value of the range
#		in which the random weights associated to the inputs of a specific neuron
#		are initialized. In the simplest case this value is common to all the
#		neurons.
#
#		5) networkList: list containing a description of the dimensions of the
#		network. Each entry corresponds to the number of neurons in a specific
#		layer. The network will have a number of layers corresponding to the
#		number of entries of the list.
#
#	RETURN VALUES:
#
#		The function returns a list of dictionaries containing the parameters of
#		the network. Note that the list has a number of dictionaries that is one
#		unit lower with respect to to the length of the networkList. This is
#		because the first layer corresponds to the input layer and is simply used
#		to model te input spikes. It is not associated to a physical layer of
#		neurons. 

def createNetworkDictList(v_th_list, v_reset, w_min_list, w_max_list, networkList):
	return [createLayerDict(v_th_list[i], v_reset, w_min_list[i], w_max_list[i], 
		networkList[i+1], networkList[i]) for i in range(len(networkList)-1)]








# Create the dictionary associated to a specific layer of the network.
#
# 	INPUT PARAMETERS:
#
# 		1) v_th: NumPy array containing the threshold voltage of each neuron in
# 		the layer.
#
# 		2) v_reset: numerical value corresponding to the voltage at which the
#		membrane potential is reset. This is common to all the neurons.
#
#		3) w_min: NumPy array. Eeach element corresponds to the minimum value of
#		the range in which the random weights associated to the inputs of a
#		specific neuron are initialized. In the simplest case this value is common
#		to all the neurons.
#
#		4) w_max: NumPy array. Eeach element corresponds to the maximum value of
#		the range in which the random weights associated to the inputs of a
#		specific neuron are initialized. In the simplest case this value is common
#		to all the neurons.
#
#		5) currLayerDim: dimension of the layer associated to the generated
#		dictionary.
#
#		6) prevLayerDim: dimension of the previous layer. This is needed to
#		dimension the arrays containing the weights and the time steps of the
#		input events.
#
#	RETURN VALUES:
#
#		layerDict: dictionary associated to the layer.

def createLayerDict(v_th, v_reset, w_min, w_max, currLayerDim, prevLayerDim):

	# Create the dictionary
	layerDict = {}

	# Initialize the membrane potential to the reset potential
	layerDict["v_mem"] = v_reset*np.ones(currLayerDim)

	# Initialize the threshold voltages
	layerDict["v_th"] = v_th

	# Initialize all the output events to False (= no spikes)
	layerDict["outEvents"] = np.zeros(currLayerDim).astype(bool)

	# Initialize the weights to random values between w_min and w_max
	layerDict["weights"] = (np.random.random((currLayerDim, prevLayerDim)).T*\
				(w_max-w_min) + w_min).T

	# Initialize the input events time steps
	layerDict["t_in"] = np.zeros(prevLayerDim).astype(int)

	# Initialize the output events time steps
	layerDict["t_out"] = np.zeros(currLayerDim).astype(int)

	return layerDict
