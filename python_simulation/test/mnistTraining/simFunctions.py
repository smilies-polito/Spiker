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
from training import updateSpikeCount,			\
			accuracyAndClassification



def trainSingleImg(poissonImg, label, labelsArray, networkDictList, v_mem_dt_tau, 
			stdp_dt_tau, v_reset, A_ltp, A_ltd, spikeCountArray,
			v_memEvol_list, outEventsEvol_list, weightsEvol_list):


	for i in range(len(poissonImg)):

		trainSnn(poissonImg[i], networkDictList, v_mem_dt_tau, stdp_dt_tau,
				v_reset, A_ltp, A_ltd, i)
		updateSpikeCount(spikeCountArray, networkDictList[-1]["outEvents"])

		# To remove once tested
		addToTemporalEvolution(networkDictList, "v_mem", v_memEvol_list, i)

		addToTemporalEvolution(networkDictList, "outEvents", 
					outEventsEvol_list, i)

		addToTemporalEvolution(networkDictList, "weights", weightsEvol_list, i)

	return accuracyAndClassification(spikeCountArray, labelsArray, label)




def addToTemporalEvolution(networkDictList, parameter, temporalEvol_list, 
				currentStep):

	for i in range(len(networkDictList)):

		temporalEvol_list[i][currentStep] = networkDictList[i][parameter]



def createSparseArray(N_sim, prevLayerDim, density):
	
	# Create a spare matrix of random values
	sparseArray = sp.random(N_sim, prevLayerDim, density = density)

	# Convert the values inside the matrix to boolean type
	sparseArray = sparseArray.A.astype(bool)

	return sparseArray



def plotTrainNetworkResults(networkList, inEvents_evolution, weightsEvol_list,
				outEventsEvol_list, v_memEvol_list,  v_th_list):


	outEventsEvol_list[0] = outEventsEvol_list[0].T
	v_memEvol_list[0] = v_memEvol_list[0].T
	weightsEvol_list[0] = np.transpose(weightsEvol_list[0], (1,2,0))

	plotTrainLayerResults(networkList[1], networkList[0], inEvents_evolution,
				weightsEvol_list[0], outEventsEvol_list[0], 
				v_memEvol_list[0], v_th_list[0])

	for i in range(1, len(networkList)-1):

		# Transpose the arrays in order to obtain the temporal evolution of
		# each neuron
		outEventsEvol_list[i] = outEventsEvol_list[i].T
		v_memEvol_list[i] = v_memEvol_list[i].T
		weightsEvol_list[i] = np.transpose(weightsEvol_list[i], (1,2,0))


		plotLayerResults(networkList[i+1], networkList[i], outEventsEvol_list[i-1],
					weightsEvol_list[0], outEventsEvol_list[i], 
					v_memEvol_list[i], v_th_list[i])



def plotTrainLayerResults(currLayerDim, prevLayerDim, inEvents_evolution, 
				weight_evolution, outEvents_evolution, v_mem_evolution, 
				v_th):

	for i in range(currLayerDim):

		plotTrainResults(prevLayerDim, inEvents_evolution, weight_evolution[i], 
				outEvents_evolution[i], v_mem_evolution[i], v_th[i])




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



