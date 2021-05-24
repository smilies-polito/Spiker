#!/Users/alessio/anaconda3/bin/python3

import sys

development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/parallelTraining"

if development not in sys.path:
	sys.path.insert(1,development)


import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np

from trainSnn import trainSnn
from training import updateSpikeCount, \
			accuracyAndClassification



# Debug version of the training on a single image.
#
# The difference with respect to the original training.py/trainSingleImage is that here
# the membrane potentials, the output spikes and the weights of each layer can be
# monitored over time. This is obtained by adding the quantities to bidimensional NumPy
# arrays step by step in order to keep trace of their temporal evolution.
#
# The function trains the network and then evaluates the result of the classification. If
# the network has correctly classified the image it returns the value 1 and leaves the
# array which associates each label to a specific neuron unchanged. If instead the network
# has failed in the classification it returns the value 0 and associates the label to the
# neuron that has generated the larger amount of spikes, changing the order of the labels
# in labelsArray.
#
# 	INPUT PARAMETERS:
#
#		1) poissonImg: bidimensional NumPy array containing the train spikes for
#		each neuron. Each element of the array corresponds to a time step. It is
#		in turns a NumPy array which contains one value (True = spike, False = no
#		spike) for each neuron.
#
#		2) label: numerical value corresponding to the number represented in the
#		input image.
# 		 
#		3) labelsArray: array containing the initial classification associated to
#		the output neurons. A good choice could be to associate each neuron to its
#		corresponding index, so neuron 0 corresponds to the label 0, neuron 1 to
#		the label 1 and so on. The association between labels and neurons is then
#		changed during the training basing on which neuron fires most when a
#		certain image is presented as an input.
#
#		4) networkDictList: list of dictionaries containing the parameters of the
#		network. See the file snnDataStruct.py for more details.
#
#		5) v_mem_dt_tau: ratio between the minimum temporal step delta_t and the
#		time constant tau of the exponential decay of the membrane potential. This
#		is common to all the neurons and affects how fast the memebrane potential
#		is decreased in absence of input spikes. The higher v_mem_dt_tau, the
#		faster the membrane potential decay.
#
#		6) stdp_dt_tau: ratio between the minimum temporal step delta_t and the
#		time constant tau of the stdp exponential. This affects the learning rate
#		of the neurons. The lower stdp_dt_tau the slower is the exponential decay
#		and the more influent become ltp and ltd.
#
#		7) v_reset: numerical value corresponding to the voltage at which the
#		membrane potential is reset. This is common to all the neurons.
#
#		8) A_ltp: constant value which multiplies the ltp exponential. This
#		affects how strong is the potentiation of the synapses through the ltp.
#
#				Delta_w = A_ltp*exp(-stdp_dt_tau)
#
#		9) A_ltd: constant value which multiplies the ltd exponential. This
#		affects how strong is the depression of the synapses through the ltd.
#
#				Delta_w = A_ltd*exp(stdp_dt_tau)
#
#		10) spikeCountArray: NumPy array containin the spike count related to the
#		considered label for each neuron in the output layer.
#
#	RETURN VALUE:
#
#		The function returns the classification result, 1 if correct, 0 if wrong.

def trainSingleImg(poissonImg, label, labelsArray, networkDictList, v_mem_dt_tau, 
			stdp_dt_tau, v_reset, A_ltp, A_ltd, spikeCountArray,
			v_memEvol_list, outEventsEvol_list, weightsEvol_list):


	# Loop over all the time steps of the image
	for i in range(len(poissonImg)):

		# Train the network on the single time step
		trainSnn(poissonImg[i], networkDictList, v_mem_dt_tau, stdp_dt_tau,
				v_reset, A_ltp, A_ltd, i)

		updateSpikeCount(spikeCountArray, networkDictList[-1]["outEvents"])


		# Add the membrane potential, output events and weights to their temporal
		# evolution
		addToTemporalEvolution(networkDictList, "v_mem", v_memEvol_list, i)

		addToTemporalEvolution(networkDictList, "outEvents", 
					outEventsEvol_list, i)

		addToTemporalEvolution(networkDictList, "weights", weightsEvol_list, i)




	return accuracyAndClassification(spikeCountArray, labelsArray, label)








# Insert a NumPy array into another NumPy array with one additional dimension.
#
# This can be used to store the values of the array step by step in order to monitor its
# temporal evolution.
#
# 	INPUT PARAMETERS:
#
# 		1) networkDictList: list of dictionaries containing the parameters of the
#		network. See the file snnDataStruct.py for more details.
#
#		2) parameter: string corresponding to the quantity to monitor. It must
#		correspond to a key present in networkDictList.
#
#		3) temporalEvol_list: list of NumPy arrays in which the specific quantity
#		is inserted. This must have one dimension more than
#		networkDictList[parameter], so if networkDictList[parameter] is
#		monodimensional it must be bidimensional and so on.
#
#		4) currentStep: index at which the monitored quantity must be inserted.
#		This allows to call the function within a loop and store the desired
#		values step by step.

def addToTemporalEvolution(networkDictList, parameter, temporalEvol_list, 
				currentStep):

	# Loop over all the layers
	for i in range(len(networkDictList)):

		# Add the desired quantity to the current step
		temporalEvol_list[i][currentStep] = networkDictList[i][parameter]










# Create a sparse boolean array.
#
# This can be used to generate a random spike train with arbitrarily sparse spikes that
# can be then used as an input for the simulation of the neural network.
#
# 	INPUT PARAMETERS:
#
# 		1) N_sim: number of simulation cycles. This corresponds to the time
# 		duration of each spikes train expressed in time steps.
#
# 		2) prevLayerDim: number of neurons in the previous layer. This is needed
# 		to generate a number of spike trains that is equal to the number of inputs
# 		of the current layer. 
#
# 		3) density: density of the spikes.
#
# 	RETURN VALUES:
#
# 		sparseArray: the function returns a bidimensional NumPy array with a
# 		number of subarrays eqaul to the number of time steps considered (N_sim).
# 		Each subarray contains the values of all the inputs in a single step. The
# 		density of the spikes considers the whole bidimensional array, so in an
# 		N_sim*N_neurons array with a density d there will be d*N_sim*N_neurons
# 		spikes.

def createSparseArray(N_sim, prevLayerDim, density):
	
	# Create a spare matrix of random values
	sparseArray = sp.random(N_sim, prevLayerDim, density = density)

	# Convert the values inside the matrix to boolean type
	sparseArray = sparseArray.A.astype(bool)

	return sparseArray







# Plot the time evolution of the input spikes, the weights, the membrane potential and the
# output spikes of each layer.
#
# This can be used to evaluate the behaviour of the network along the training by
# graphically observe its evolution.
#
# 	INPUT PARAMETERS: 
#
# 		1) networkList: list containing a description of the dimensions of the
#		network. Each entry corresponds to the number of neurons in a specific
#		layer. The network will have a number of layers corresponding to the
#		number of entries of the list.
#
#		2) inEvents_evolution: bidimensional NumPy array containing the input
#		spike trains with the correct shape for the plot.
#
#		3) weightsEvol_list: list of tridimensional NumPy arrays. Each entry of
#		the list corresponds to a layer. Each tridimensional array contains the
#		temporal evolution of the weights of each neuron in the layer.
#
#		4) outEventsEvol_list: list of bidimensional NumPy arrays. Each entry
#		corresponds to a layer. Each bidimensional array contains the
#		temporal evolution of the output spikes generated by each neuron in the
#		layer.
#
#		5) v_memEvol_list: list of bidimensional NumPy arrays. Each entry
#		corresponds to a layer. Each bidimensional array contains the
#		temporal evolution of the membrane potential of each neuron in the
#		layer.
#
#		6) v_th_list: list of NumPy arrays. Each entryvcorresponds to a layer.
#		Each array contains the value of the threshold for each neuron in the
#		layer.
#
# The temporal evolution of the membrane potentials, output spikes and weights is supposed
# to be stored in form of NumPy arrays in which each entry corresponds to a time step. It
# is more convenient represent them as NumPy array in which each entry corresponds to the
# temporal evolution of a specific neuron. For this reason the three arrays are properly
# transposed before plotting them.

def plotTrainNetworkResults(networkList, inEvents_evolution, weightsEvol_list,
				outEventsEvol_list, v_memEvol_list,  v_th_list):


	# Transpose the arrays to prepare them for the plot
	outEventsEvol_list[0] = outEventsEvol_list[0].T
	v_memEvol_list[0] = v_memEvol_list[0].T
	weightsEvol_list[0] = np.transpose(weightsEvol_list[0], (1,2,0))

	# Plot the temporal evolution of the first layer, using the input spikes as an
	# input
	plotTrainLayerResults(networkList[1], networkList[0], inEvents_evolution,
				weightsEvol_list[0], outEventsEvol_list[0], 
				v_memEvol_list[0], v_th_list[0])

	# Loop over all the other layers
	for i in range(1, len(networkList)-1):

		# Transpose the arrays to prepare them for the plot
		outEventsEvol_list[i] = outEventsEvol_list[i].T
		v_memEvol_list[i] = v_memEvol_list[i].T
		weightsEvol_list[i] = np.transpose(weightsEvol_list[i], (1,2,0))


		# Plot the temporal evolution of the first layer, using the output spikes
		# of the previous layer as an input.
		plotLayerResults(networkList[i+1], networkList[i], outEventsEvol_list[i-1],
					weightsEvol_list[0], outEventsEvol_list[i], 
					v_memEvol_list[i], v_th_list[i])




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

	# Loop over all the neurons in the layer
	for i in range(currLayerDim):

		plotTrainResults(prevLayerDim, inEvents_evolution, weight_evolution[i], 
				outEvents_evolution[i], v_mem_evolution[i], v_th[i])




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





# Interleave the plots of two bidimensional arrays.
#
# This can be used to alternate the plots of input spikes and weights in order to evaluate
# the temporal evolution of the weights with respect to the arrive time of the input
# spikes.
#
# 	INPUT PARAMETERS;
#
# 		1) axs: preallocate subolot object. This must be properly dimensioned to host
# 		all the subplots and can be created through a call to
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
