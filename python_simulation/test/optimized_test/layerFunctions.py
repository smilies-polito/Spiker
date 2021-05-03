#!/Users/alessio/anaconda3/bin/python3

import sys

development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/optimized_test"

if development not in sys.path:
	sys.path.insert(1,development)

from layer import layer
from neuronFunctions import createNeuronDict, plotResults


# Simulate the layer. The function loops over all the neurons in the layer and
# updates both the membrane potentials and the events generated by each neuron,
# storing their temporal evolution inside proper arrays.
#
# INPUT PARAMETERS:
# 
# 	1) N_sim: desired number of simulation cycles.
#
# 	2) inEvents_evolution: bidimensional NumPy array which contains on each
# 	   row the input events of each neuron in the layer in the specific step.
# 	   The rows represent the temporal evolution of these events.
#
# 	3) neuronDictList: list of dictionaries containing all the parameters of
# 	   the neurons.
#
# 	4) dt_tau: ratio delta_t/tau. See the function "neuron" for more details.
#
# 	5) outEvents_evolution: preallocated bidimensional NumPy array which is
# 	   filled by the function with the events generated by each neuron at
# 	   every simulation step.
#
# 	6) v_mem_evolution: preallocated bidimensional NumPy array which is
# 	   filled by the function with the membrane potential of each neuron at
# 	   every simulation step.
#
# 	7) currLayerDim: number of neurons in the current layer. Needed by the
# 	   function "layer" to dimension of the updating loop.

def simulateLayer(N_sim, inEvents_evolution, neuronsDictList, dt_tau, 
		outEvents_evolution, v_mem_evolution, currLayerDim):

	for i in range(N_sim):

		# Update the layer
		layer(inEvents_evolution[i], neuronsDictList, dt_tau, 
			outEvents_evolution[i], currLayerDim)

		# Store the membrane potentials
		storeMembranePotentials(currLayerDim, neuronsDictList, v_mem_evolution[i])




# Create a list of dictionaries containing the parameters of each neuron.
#
# INPUT PARAMETERS:
#
# 	1) currLayerDim: number of neurons in the current layer. Needed to dimension
# 	   the list.
#
# 	2) v_th, v_res, w_min and w_max: NumPy array containing the desired
# 	   parameters of each neuron. 
#
# 	3) prevLayerDim: number of neurons in the previous layer. Needed to dimension
# 	   the array of weights in each neuron.
#
# See createNeuronDict for more details on the input parameters.

def createDictList(currLayerDim, v_th, v_res, prevLayerDim, w_min, w_max):

	return [createNeuronDict(v_th[i], v_res[i], prevLayerDim, w_min[i], w_max[i])
		for i in range(currLayerDim)]





# Plot the results of each neuron in the layer.
#
# INPUT PARAMETERS:
#
# 	1) currLayerDim: number of neurons in the current layer. Needed to
# 	   determine the number of plots to represent.
#
# 	2) prevLayerDim: number of neurons in the previous layer. Needed to
# 	   structure the plots, dividing them in subplots.
#
# 	3) inEvents_evolution: bidimensional NumPy array containing the
# 	   temporal evolution of the input events coming from the previous layer.
# 	   The events are the same for each neuron, being the network fully
# 	   connected.
#
# 	4) outEvents_evolution, v_mem_evolution: bidimensional NumPy arrays
# 	   containing the temporal evolution of outEvents and v_mem for each
# 	   neuron in the current layer.
#
# 	5) v_th: NumPy array containing the value of the threshold for each
#	   neuron.
#
# See plotResults for more details on the input parameters.

def plotLayerResults(currLayerDim, prevLayerDim, inEvents_evolution, outEvents_evolution, 
			v_mem_evolution, v_th):

	for i in range(currLayerDim):

		plotResults(prevLayerDim, inEvents_evolution, outEvents_evolution[i], 
			v_mem_evolution[i], v_th[i])



# Store the membrane potential of all the neurons in the layer inside a NumPy array.
#
# INPUT PARAMETERS:
#
# 	1) currLayerDim: number of neurons in the current layer. Needed to
# 	   dimension the loop used to store all the values.
#
# 	2) neuronsDictList: list of dictionaries which contain the parameters of all
# 	   the neurons inside the layer.
#
# 	3) v_mem_array: NumPy array that will be updated with the membrane potentials
# 	   of all the neurons. It is overwritten by the function, so it can be
# 	   initialized to whatever values.

def storeMembranePotentials(currLayerDim, neuronsDictList, v_mem_array):

	for i in range(currLayerDim):
		v_mem_array[i] = neuronsDictList[i]["v_mem"]
