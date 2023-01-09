#!/Users/alessio/anaconda3/bin/python3

import sys

development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/parallelTest"

if development not in sys.path:
	sys.path.insert(1,development)

from snn import updateNetwork
from layerFunctions import createLayerDict, \
			plotLayerResults

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


# Simulate a network of spiking neurons for the desired number of simulation cycles.
#
# INPUT PARAMETERS:
#
# 	1) N_sim: number of simulation steps.
#
# 	2) inEvents_evolution: bidimensional NumPy array which contains the temporal
# 	   evolution of the input spikes.
#
# 	3) networkDictList: list of dictionaries which contain the parameters of each
# 	   layer.
#
# 	4) outEvents_evolution: bidimensional NumPy array which is filled by the
# 	   function with the temporal evolution of the output events of each layer.
#
# 	5) v_mem_evolution: bidimensional NumPy array which is filled by the
# 	   function with the temporal evolution of the membrane potentials.
#
# 	6) dt_tau, v_reset: numerical values. These are parameters that are common to
# 	   all the neurons in the layer. See updateLayer for more details.


def simulateSnn(N_sim, inEvents_evolution, networkDictList, outEventsEvol_list, 
		v_memEvol_list, dt_tau, v_reset):

	for i in range(N_sim):
	
		updateNetwork(inEvents_evolution[i], networkDictList, dt_tau, v_reset)

		addToTemporalEvolution(networkDictList, "v_mem", v_memEvol_list, i)

		addToTemporalEvolution(networkDictList, "outEvents", 
					outEventsEvol_list, i)



# Create the list of dictionaries to store the parameters of all the layers.
#
# INPUT PARAMETERS:
#
# 	1) v_th_list: list of NumPy arrays. Each array contains the threshold voltages of 
# 	   all the neurons in a specific layer.
#
# 	2) v_reset: numerical value representing the voltage at which the membrane
# 	   potential of each neuron is reset when it exceeds the threshold.
#
# 	3) w_min_list, w_max_list: lists of NumPy arrays. Each array contains the minimum 
# 	   (or maximum) ends of the range in which the random weights must be initialized
# 	   for each neuron in a specific layer.
#
# 	4) networkList: list in which each entry corresponds to the number of nodes inside
# 	   the layer.

def createNetworkDictList(v_th_list, v_reset, w_min_list, w_max_list, networkList):
	return [createLayerDict(v_th_list[i], v_reset, w_min_list[i], w_max_list[i], networkList[i+1], 
		networkList[i]) for i in range(len(networkList)-1)]




# Add the current step data to the temporal evolution.
#
# INPUT PARAMETERS:
#
# 	1) networkDictList: list of dictionaries containg the parameters of each layer.
#
# 	2) parameter: string which identifies the parameter to consider.
#
# 	2) temporalEvol_list: list of bidimensional NumPy arrays containing the temporal
# 	   evolution of the same parameter or output variable of currentStep_list.
#
# 	3) currentStep: current simulation step. Working with a simulation loop this
# 	   corresponds to the current value of the simulation index.
#
# The function adds the current value of the required element of each layer to the list 
# of temporal evolutions. This allows to monitor the desired quantity, so for example the 
# membrane potential or the output spikes of each layer, and to visualize its temporal 
# evolution.

def addToTemporalEvolution(networkDictList, parameter, temporalEvol_list, 
				currentStep):

	for i in range(len(networkDictList)):

		temporalEvol_list[i][currentStep] = networkDictList[i][parameter]





# Plot the results of each neuron in the network.
#
# INPUT PARAMETERS:
#
# 	1) networkList: list which describes the net. It contains one entry for each layer.
# 	   Each entry is an integer number which represents the number of nodes in the
# 	   layer. It is important that this list includes also the the input layer in
# 	   order to properly organize the plots.
#
# 	2) inEvents_evolution: bidimensional NumPy array containing the
# 	   temporal evolution of the input events coming from the previous layer.
# 	   The events are the same for each neuron, being the network fully
# 	   connected.
#
# 	3) outEventsEvol_list, v_memEvol_list: list of bidimensional NumPy arrays
# 	   containing the temporal evolution of the output events and the membrane 
# 	   potential for each neuron in each layer. The arrays stored in the list are 
# 	   expected to be ordered in the same way in which the functions 
# 	   storeOutputEvolution and storeNetworkPotentials stored them. 
# 	   This means that each row corresponds to a simulation step and contains the 
# 	   values of the output or the membrane potential for each neuron in the layer. 
# 	   In order to correctly plot them the arrays need to be transposed. In this way 
# 	   each row contains the temporal evolution of a specific neuron in the layer. 
# 	   For this reason the function transpose each array before plotting it.
#
# 	4) v_th_list: list of NumPy arrays containing the value of the threshold for each
#	   neuron.
#
# See plotLayerResults for more details on the input parameters.

def plotNetworkResults(networkList, inEvents_evolution, outEventsEvol_list, 
			v_memEvol_list, v_th_list):


	outEventsEvol_list[0] = outEventsEvol_list[0].T
	v_memEvol_list[0] = v_memEvol_list[0].T

	plotLayerResults(networkList[1], networkList[0], inEvents_evolution,
				outEventsEvol_list[0], v_memEvol_list[0], v_th_list[0])

	for i in range(1, len(networkList)-1):

		# Transpose the arrays in order to obtain the temporal evolution of
		# each neuron
		outEventsEvol_list[i] = outEventsEvol_list[i].T
		v_memEvol_list[i] = v_memEvol_list[i].T

		plotLayerResults(networkList[i+1], networkList[i], outEventsEvol_list[i-1],
				outEventsEvol_list[i], v_memEvol_list[i], v_th_list[i])
