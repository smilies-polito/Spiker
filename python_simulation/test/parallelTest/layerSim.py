#!/Users/alessio/anaconda3/bin/python3

import numpy as np

# Create the dictionary that contains all the parameters of the layer.
#
# INPUT PARAMETERS:
#
# 	1) v_th: NumPy array containing the threshold of each neuron.
#
# 	2) v_reset: numerical value at which the membrane potential is reset at the
# 	   beginning and whenever it exceeds the threshold.
#
# 	3) w_min and w_max: NumPy arrays containing the minimum and the maximum of the 
# 	   range in which the random weights will be generated. Working with NumPy
# 	   arrays with a dimension equal to the number of nodes in the layer each
# 	   neuron can have its own range.
#
# 	4) currLayerDim: dimension of the layer to create.
#
# 	5) prevLayerDim: dimension of the previous layer of neurons. Needed to dimension
# 	   the array of weights

def createLayerDict(v_th, v_reset, w_min, w_max, currLayerDim, prevLayerDim):

	layerDict = {}

	layerDict["v_mem"] = v_reset*np.ones(currLayerDim)

	layerDict["v_th"] = v_th

	layerDict["outEvents"] = np.zeros(currLayerDim)

	layerDict["weights"] = (np.random.random((currLayerDim, prevLayerDim)).T*\
				(w_max-w_min) + w_min).T

	return layerDict
