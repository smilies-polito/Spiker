#!/Users/alessio/anaconda3/bin/python3

import sys

development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/parallelTraining"

if development not in sys.path:
	sys.path.insert(1,development)

from trainLayer import trainLayer

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


# Simulate the evolution of the layer.
#
# INPUT PARAMETERS:
#
#	1) N_sim: number of simulation steps.
#
# 	2) inEvents_evolution: bidimensional NumPy array containing the temporal 
# 	   evolution of the input events. Each row contains all the input events,
# 	   provided by the external world or by a previous layer of neurons in a specific 
# 	   time step.
#
# 	3) layerDict: dictionary containing all the parameters of the layer.
#
# 	4) outEvents_evolution: bidimensioal NumPy array that is filled by the function 
# 	   with the temporal evolution of the outputs of the layer.
#
# 	5) v_mem_evolution: bidimensional NumPy array that is filled by the function
# 	   with the temporal evolution of the membrane potentials of the layer.
#
# 	6) v_mem_dt_tau, stdp_dt_tau, v_reset: numerical values. These are parameters 
# 	   that are common to all the neurons in the layer. See updateLayer for more 
# 	   details.

def simulateTrainLayer(N_sim, inEvents_evolution, layerDict, weights_evolution,
			outEvents_evolution, v_mem_evolution, v_mem_dt_tau, 
			stdp_dt_tau, v_reset, A_ltp, A_ltd):

	for i in range(N_sim):
		
		trainLayer(inEvents_evolution[i], layerDict, v_mem_dt_tau, 
				stdp_dt_tau, v_reset, A_ltp, A_ltd, i)

		outEvents_evolution[i] = layerDict["outEvents"]
		v_mem_evolution[i] = layerDict["v_mem"]
		weights_evolution[i] = layerDict["weights"]	



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
# 	   the array of weights and input time instants.

def createLayerDict(v_th, v_reset, w_min, w_max, currLayerDim, prevLayerDim):

	layerDict = {}

	layerDict["v_mem"] = v_reset*np.ones(currLayerDim)

	layerDict["v_th"] = v_th

	layerDict["outEvents"] = np.zeros(currLayerDim).astype(bool)

	layerDict["weights"] = (np.random.random((currLayerDim, prevLayerDim)).T*\
				(w_max-w_min) + w_min).T

	layerDict["t_in"] = np.zeros(prevLayerDim).astype(int)

	layerDict["t_out"] = np.zeros(currLayerDim).astype(int)

	return layerDict




# Create a sparse bidimensional array. This can be used to create the input spikes that 
# will be provided to the network
#
# INPUT PARAMETERS:
#
# 	1) N_sim: number of simulation cycles
#
# 	2) prevLayerDim: dimension of the previous layer
#
#	3) density: density of spikes in the whole bidimensional
#	   array
#
# The function returns a N_sim x prevLayerDim NumPy array

def createSparseArray(N_sim, prevLayerDim, density):
	
	# Create a spare matrix of random values
	sparseArray = sp.random(N_sim, prevLayerDim, density = density)

	# Convert the values inside the matrix to boolean type
	sparseArray = sparseArray.A.astype(bool)

	return sparseArray






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

def plotTestLayerResults(currLayerDim, prevLayerDim, inEvents_evolution, 
			outEvents_evolution, v_mem_evolution, v_th):

	for i in range(currLayerDim):

		plotTestResults(prevLayerDim, inEvents_evolution, outEvents_evolution[i], 
			v_mem_evolution[i], v_th[i])






# Plot the time evolution of the input spikes, the weights, the membrane potential and the
# output spikes for all the neurons in a single layer.
#
# The function creates one plot for each neuron in the layer.
#
# 	INPUT PARAMETERS:
#
# 		1) currLayerDim: dimension of the current layer. Needed to determine the
# 		total amount of plots, one for each neuron.
#
#		2) prevLayerDim: dimension of the previous layer. needed to determine the
#		number of subplots inside the plot of each layer.
#
#		3) inEvents_evolution: bidimensional NumPy array containing the input
#		spike trains with the correct shape for the plot.
#
#		4) weights_evolution: tridimensional NumPy array containing the temporal
#		evolution of the weights of each neuron with the correct shape for the
#		plot.
#
#		5) outEvents_evolution: bidimensional NumPy array containing the temporal
#		evolution of the output spikes  with the correct shape for the plot.
#
#		6) v_mem_evolution: bidimensional NumPy array containing the temporal
#		evolution of the membrane potentials with the correct shape for the plot.
#
#		7) v_th: Numpy array containin the value of the threshold for each neuron
#		in the layer. 


def plotTrainLayerResults(currLayerDim, prevLayerDim, inEvents_evolution, 
				weight_evolution, outEvents_evolution, v_mem_evolution, 
				v_th):

	for i in range(currLayerDim):

		plotTrainResults(prevLayerDim, inEvents_evolution, weight_evolution[i], 
				outEvents_evolution[i], v_mem_evolution[i], v_th[i])





# Function to plot input events, membrane potential and output events of a 
# neuron on a single plot organized in subplots
#
# INPUT PARAMETERS:
#
# 	1) prevLayerDim: number of neurons in the previous layer. This is
# 	   needed to dimension the number of subplots. They need to be
# 	   prevLayerDim + 2 in order to include also the subplots of the
# 	   membrane potential and of the output events.
#
# 	2) inEvents: NumPy bidimensional array containing the temporal
# 	   evolution of the spikes generated by each neuron in the previous
# 	   layer.
#
# 	3) outEvents: NumPy array which contains the temporal evolution of
# 	   the events generated by the current neuron.
#
# 	4) v_mem_evolution: NumPy array which contains the temporal evolution
# 	   of the membrane potential.
#
# 	5) v_th: firing threshold. It will be represented with a dashed line
# 	   together with the membrane potential in order to clearly visualize
# 	   when the threshold is exceeded.

def plotTestResults(prevLayerDim, inEvents, outEvents, v_mem_evolution, v_th):

	fig, axs = plt.subplots(prevLayerDim+2, 1)

	# Plot the input events
	subplotMultipleArrays(axs, inEvents, "Input Spikes")

	# Plot the membrane potential and the threshold
	subplotArrayAndConst(axs[prevLayerDim], v_mem_evolution, "Membrane Potential", 
				v_th)

	# Plot the output events
	subplotArray(axs[prevLayerDim+1], outEvents, "Output Spikes")

	# Increase the spacing between subplots
	plt.subplots_adjust(hspace = 1.5)

	# Show the plot
	plt.show()






# Plot the time evolution of the input spikes, the weights, the membrane potential and the
# output spikes for a single neuron.
#
# 	INPUT PARAMETERS:
#
# 		1) prevLayerDim: dimension of the previous layer. needed to determine the
#		number of subplots inside the plot of each layer.
#
#		2) inEvents_evolution: bidimensional NumPy array containing the input
#		spike trains with the correct shape for the plot.
#
#		3) weights_evolution: bidimensional NumPy array containing the temporal
#		evolution of the weights of the specific neuron with the correct shape for
#		the plot.
#
#		4) outEvents_evolution: NumPy array containing the temporal evolution of
#		the output spikes generated by the neuron with the correct shape for the
#		plot.
#
#		5) v_mem_evolution: NumPy array containing the temporal evolution of the
#		membrane potential of the neuron with the correct shape for the plot.
#
#		6) v_th: threshold of the neuron.

def plotTrainResults(prevLayerDim, inEvents_evolution, weight_evolution, 
			outEvents_evolution, v_mem_evolution, v_th):

	fig, axs = plt.subplots(2*prevLayerDim+2, 1)

	# Plot the input events
	interleaveArraySubplots(axs, inEvents_evolution, weight_evolution, "Input Spike",
					"Weight")

	# Plot the membrane potential and the threshold
	subplotArrayAndConst(axs[2*prevLayerDim], v_mem_evolution, "Membrane Potential", 
				v_th)

	# Plot the output events
	subplotArray(axs[2*prevLayerDim+1], outEvents_evolution, "Output Spikes")

	# Increase the spacing between subplots
	plt.subplots_adjust(hspace = 1.5)

	# Show the plot
	plt.show()





# Plot a generic array on a subplot.
#
# INPUT PARAMETERS:
#
# 	1) axs: subplot object on which the array will be plotted. This must
# 	   have been previously created using matplotlib.pyplot.subplots()
# 	   function.
#
# 	2) plotData: NumPy array containing the values to plot
#
# 	3) title: string which contains the title to assign to the plot

def subplotArray(axs, plotData, title):

	axs.plot(plotData)
	axs.grid()
	axs.set_xticks(np.arange(0, plotData.size, 
			step = plotData.size/20))
	axs.set_title(title)






# Plot the desired number of arrays on an equivalent number of subplots.
#
# INPUT PARAMETERS:
#
# 	1) axs: subplot object on which the arrays will be plotted. This must
# 	   have been previously created using matplotlib.pyplot.subplots()
# 	   function with dimensions that are consistent with the number of
# 	   subplots needed
#
# 	2) plotData: NumPy bidimensional array containing all the arrays to plot
#
# 	3) title: string which contains the generic title of the group of
# 	   subplots

def subplotMultipleArrays(axs, plotData, title):

	# Compute the total amount of subplots
	N_subplots = plotData.shape[0]

	axs[0].set_title(title)

	for i in range(N_subplots):
		axs[i].plot(plotData[i])
		axs[i].grid()
		axs[i].set_xticks(np.arange(0, plotData[i].size, 
				step = plotData[i].size/20))






# Interleave the plots of two bidimensional arrays.
#
# This can be used to alternate the plots of input spikes and weights in order to evaluate
# the temporal evolution of the weights with respect to the arrive time of the input
# spikes.
#
# 	INPUT PARAMETERS;
#
# 		1) axs: preallocate subolot object. This must be properly dimensioned to 
# 		host all the subplots and can be created through a call to
# 		matplotlib.pyplot.subplots.
#
#		2) plotData1: first bidimensional NumPy array containing N arrays to plot.
#
#		3) plotData2: second bidimensional NumPy array containing N arrays to plot.
#
#		4) title1: string used as the base for the title of the plots of the first
#		array. The number of the subplot will be appended to this title.
#
#		5) title2: string used as the base for the title of the plots of the
#		second array. The number of the subplot will be appended to this title.
#
# The function will fill 2*N axs subplots, where N is the number of elements of plotData1,
# equal to the number of elements of plotData2.


def interleaveArraySubplots(axs, plotData1, plotData2, title1, title2):

	# Compute the total amount of subplots for one of the two arrays
	N_subplots = plotData1.shape[0]

	for i in range(0, 2*N_subplots, 2):

		# First array
		title1_i = title1 + " " + str(int(i/2))
		axs[i].set_title(title1_i)
		axs[i].plot(plotData1[int(i/2)])
		axs[i].grid()
		axs[i].set_xticks(np.arange(0, plotData1[int(i/2)].size, 
				step = plotData1[int(i/2)].size/20))

		# Second array
		title2_i = title2 + " " + str(int(i/2))
		axs[i+1].set_title(title2_i)
		axs[i+1].plot(plotData2[int(i/2)])
		axs[i+1].grid()
		axs[i+1].set_xticks(np.arange(0, plotData2[int(i/2)].size, 
				step = plotData2[int(i/2)].size/20))




	
# Plot a generic array on a subplot together with a constant dashed line
#
# INPUT PARAMETERS:
#
# 	1) axs: subplot object on which the array will be plotted. This must
# 	   have been previously created using matplotlib.pyplot.subplots()
# 	   function.
#
# 	2) plotData: NumPy array containing the values to plot
#
# 	3) title: string which contains the title to assign to the plot
#
# 	4) constValue: value that will be plot together with the array

def subplotArrayAndConst(axs, plotData, title, constValue):

	constValue = constValue*np.ones(plotData.size)

	axs.plot(plotData)
	axs.plot(constValue, "--")
	axs.grid()
	axs.set_xticks(np.arange(0, plotData.size, 
			step = plotData.size/20))
	axs.set_title(title)

