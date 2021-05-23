#!/Users/alessio/anaconda3/bin/python3

import sys

development = "/Users/alessio/Documents/Poli/Magistrale/Tesi/\
Tesi/spiker/python_simulation/development/parallelTraining"

if development not in sys.path:
	sys.path.insert(1,development)


from trainSnn import trainSnn
from poisson import imgToSpikeTrain

import numpy as np


# Train the network over a set of images and labels and print intermediate results.
#
# The function splits the dataset into the desired number of subsets and then trains the
# network printing the intermediate results at the end of each subset.
#
# 	INPUT PARAMETERS:
#
# 		1) images: bidimensional NumPy array containing the entire set of images.
# 		Each element of the array, in turns a NumPy array, corresponds to a black
# 		and white image and contains the integer value of each pixel, representing
# 		its position in a gray scale. For the MNIST 0 is totally black, 255 is
# 		totally white.
#
# 		2) labels: NumPy array of integers containing the value associated to
# 		each image.
#
#		3) N_subsets: desired number of subsets in which the dataset is divided.
#
#		4) timeEvolCycles: duration of the spike train in which each pixel is
#		encoded expressed in time steps.
#
#		5) N_pixels: number of pixels that compose each image.
#
#		6) pixelMin: minimum value of the pixel. For the MNIST this is 0 and
#		corresponds to a complete black.
#
#		7) pixelMax: maximum value of the pixel. For the MNIST this is 255 and
#		corresponds to a complete white.
#
#		8) labelsArray: array containing the initial classification associated to
#		the output neurons. A good choice could be to associate each neuron to its
#		corresponding index, so neuron 0 corresponds to the label 0, neuron 1 to
#		the label 1 and so on. The association between labels and neurons is then
#		changed during the training basing on which neuron fires most when a
#		certain image is presented as an input.
#
#		9) networkDictList: list of dictionaries containing the parameters of the
#		network. See the file snnDataStruct.py for more details.
#
#		10) v_mem_dt_tau: ratio between the minimum temporal step delta_t and the
#		time constant tau of the exponential decay of the membrane potential. This
#		is common to all the neurons and affects how fast the memebrane potential
#		is decreased in absence of input spikes. The higher v_mem_dt_tau, the
#		faster the membrane potential decay.
#
#		11) stdp_dt_tau: ratio between the minimum temporal step delta_t and the
#		time constant tau of the stdp exponential. This affects the learning rate
#		of the neurons. The lower stdp_dt_tau the slower is the exponential decay
#		and the more influent become ltp and ltd.
#
#		12) v_reset: numerical value corresponding to the voltage at which the
#		membrane potential is reset. This is common to all the neurons.
#
#		13) A_ltp: constant value which multiplies the ltp exponential. This
#		affects how strong is the potentiation of the synapses through the ltp.
#
#				Delta_w = A_ltp*exp(-stdp_dt_tau)
#
#		14) A_ltd: constant value which multiplies the ltd exponential. This
#		affects how strong is the depression of the synapses through the ltd.
#
#				Delta_w = A_ltd*exp(stdp_dt_tau)
#
#		15) classificationArray: bidimensional Numpy array containing one array
#		per label. Each array contains the count of the spikes emitted by each
#		neuron from the beginning of the training when presented with the image
#		corresponding to the specific label. This is used to evaluate the
#		classification performances and to modify labelsArray choosing the most
#		firing neuron for each label.

def train(images, labels, N_subsets, timeEvolCycles, N_pixels, pixelMin, pixelMax,
		labelsArray, networkDictList, v_mem_dt_tau, stdp_dt_tau, v_reset, A_ltp, 
		A_ltd, classificationArray):

		# Divide the dataset into the desired amount of subsets
		subsetLength = int(images.shape[0] // N_subsets)
		lastSubsetLength = images.shape[0] % N_subsets

		if(lastSubsetLength) == 0:
			lastSubsetLength = subsetLength

		# Loop over the first N-1 subsets
		for i in range(N_subsets-1):
			
			# Train the network over the subset
			accuracy = trainSubset(images[subsetLength*i:subsetLength*(i+1)], 
						labels[subsetLength*i:subsetLength*(i+1)], 
						subsetLength, timeEvolCycles, N_pixels, 
						pixelMin, pixelMax, labelsArray, 
						networkDictList, v_mem_dt_tau, 
						stdp_dt_tau, v_reset, A_ltp, A_ltd, 
						classificationArray)

			# Print the intermediate results
			logString = "Accuracy: " + str(accuracy) + ", Labels' array: " + \
					str(labelsArray)
			print(logString)

		# Train the network over the last subset
		accuracy = trainSubset(images[subsetLength*(N_subsets-1):
					subsetLength*(N_subsets-1) + lastSubsetLength], 
					labels[subsetLength*(N_subsets-1):
					subsetLength*(N_subsets-1) + lastSubsetLength], 
					subsetLength, timeEvolCycles, N_pixels, 
					pixelMin, pixelMax, labelsArray, 
					networkDictList, v_mem_dt_tau, 
					stdp_dt_tau, v_reset, A_ltp, A_ltd, 
					classificationArray)

		# Print the final results
		logString = "Accuracy: " + str(accuracy) + ", Labels' array: " + \
					str(labelsArray)
		print(logString)









# Train the network over a subset of images and labels.
#
# The function trains the network over the desired number of images and labels and returns
# the classification accuracy. The subset could also be composed by the whole dataset. 
#
# 	INPUT PARAMETERS:
#
# 		
# 		1) images: bidimensional NumPy array containing the entire set of images.
# 		Each element of the array, in turns a NumPy array, corresponds to a black
# 		and white image and contains the integer value of each pixel, representing
# 		its position in a gray scale. For the MNIST 0 is totally black, 255 is
# 		totally white.
#
# 		2) labels: NumPy array of integers containing the value associated to
# 		each image.
#
#		3) subsetLength: desired amount of images in the training set.
#
#		4) timeEvolCycles: duration of the spike train in which each pixel is
#		encoded expressed in time steps.
#
#		5) N_pixels: number of pixels that compose each image.
#
#		6) pixelMin: minimum value of the pixel. For the MNIST this is 0 and
#		corresponds to a complete black.
#
#		7) pixelMax: maximum value of the pixel. For the MNIST this is 255 and
#		corresponds to a complete white.
#
#		8) labelsArray: array containing the initial classification associated to
#		the output neurons. A good choice could be to associate each neuron to its
#		corresponding index, so neuron 0 corresponds to the label 0, neuron 1 to
#		the label 1 and so on. The association between labels and neurons is then
#		changed during the training basing on which neuron fires most when a
#		certain image is presented as an input.
#
#		9) networkDictList: list of dictionaries containing the parameters of the
#		network. See the file snnDataStruct.py for more details.
#
#		10) v_mem_dt_tau: ratio between the minimum temporal step delta_t and the
#		time constant tau of the exponential decay of the membrane potential. This
#		is common to all the neurons and affects how fast the memebrane potential
#		is decreased in absence of input spikes. The higher v_mem_dt_tau, the
#		faster the membrane potential decay.
#
#		11) stdp_dt_tau: ratio between the minimum temporal step delta_t and the
#		time constant tau of the stdp exponential. This affects the learning rate
#		of the neurons. The lower stdp_dt_tau the slower is the exponential decay
#		and the more influent become ltp and ltd.
#
#		12) v_reset: numerical value corresponding to the voltage at which the
#		membrane potential is reset. This is common to all the neurons.
#
#		13) A_ltp: constant value which multiplies the ltp exponential. This
#		affects how strong is the potentiation of the synapses through the ltp.
#
#				Delta_w = A_ltp*exp(-stdp_dt_tau)
#
#		14) A_ltd: constant value which multiplies the ltd exponential. This
#		affects how strong is the depression of the synapses through the ltd.
#
#				Delta_w = A_ltd*exp(stdp_dt_tau)
#	
#		15) classificationArray: bidimensional Numpy array containing one array
#		per label. Each array contains the count of the spikes emitted by each
#		neuron from the beginning of the training when presented with the image
#		corresponding to the specific label. This is used to evaluate the
#		classification performances and to modify labelsArray choosing the most
#		firing neuron for each label.
#
#	RETURN VALUE:
#		
#		The function returns the classification accuracy obtained during the
#		training.

def trainSubset(images, labels, subsetLength, timeEvolCycles, N_pixels, pixelMin, pixelMax,
		labelsArray, networkDictList, v_mem_dt_tau, stdp_dt_tau, v_reset, A_ltp, 
		A_ltd, classificationArray):

		# Initialize the accuracy before starting the training
		accuracy = 0

		# Loop over all the images in the subset
		for i in range(subsetLength):

			# Convert the image in spikes trains
			poissonImg = imgToSpikeTrain(images[i], timeEvolCycles, N_pixels,
					pixelMin, pixelMax)

			# Select the label corresponding to the image
			label = labels[i]

			# Train the network on a single image and return the
			# classification result
			classResult = trainSingleImg(poissonImg, label, labelsArray,
					networkDictList, v_mem_dt_tau, stdp_dt_tau,
					v_reset, A_ltp, A_ltd, 
					classificationArray[label])

			# Update the accuracy
			accuracy = accuracy + classResult

			# Bring the network to a quiet state in order to start a new
			# training session with a new image
			restartNetwork(networkDictList)	

		return accuracy/subsetLength





# Train the network om a single image and label couple.
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
			stdp_dt_tau, v_reset, A_ltp, A_ltd, spikeCountArray):


	# Loop over all the temporal steps
	for i in range(len(poissonImg)):

		# Train the network on the specific step
		trainSnn(poissonImg[i], networkDictList, v_mem_dt_tau, stdp_dt_tau,
				v_reset, A_ltp, A_ltd, i)

		updateSpikeCount(spikeCountArray, networkDictList[-1]["outEvents"])

	# Modify the order of the labels if necessary and returns the classification
	# result
	return accuracyAndClassification(spikeCountArray, labelsArray, label)







# Update the spike count related to the considered label for each enuron in the output
# layer.
#
# 	INPUT PARAMETERS:
#
# 		1) spikeCountArray: NumPy array containing one entry for each output
# 		neuron. The array corresponds to a specific label.
#
# 		2) outEvents: NumPy array containing the events generated by the output
# 		layer. This is used to address spikeCountArray and update the spike count
# 		only for the neurons that have generated an event

def updateSpikeCount(spikeCountArray, outEvents):
	spikeCountArray[outEvents] +=1




# Determine whether the classification of the network is correct or not. If not change the
# order of the labels in labelsArray associating the label of the current image to the
# neuron that has generated the larger amount of spikes.
#
# 	INPUT PARAMETERS:
#
# 		1) spikeCountArray: Numpy array which contains the count of the spikes 
# 		emitted by each neuron in correspondence of the	current label.
#
# 		2) labelsArray: NumPy array which contains the labels orderer in a way
# 		that associates each label to the neuron that has generated the larger
# 		amount of spikes for it.
#
# 		3) label: current value of the label associated to the input image.
#
# 	RETURN VALUES:
#
# 		0 if the classification is wrong
#
# 		1 if the classification is right

def accuracyAndClassification(spikeCountArray, labelsArray, label):

	if label in labelsArray[spikeCountArray == np.max(spikeCountArray)]:
		return 1

	else:
		updateClassification(spikeCountArray, labelsArray, label)
		return 0




# Update the order of the labels in order to associate to the current label the neuron
# that has generate the largest amount of spikes for it. 
#
# The function switches two elements of the labels' array. It puts the value of the
# current label in the position corresponding to the most firing neuron. The previous
# value associated to the neuron is moved to the position in which the current label's
# value was stored.
#
# 	INPUT PARAMETERS:
#
# 		1) spikeCountArray: NumPy array containing the count of the spikes emitted
# 		by each neuron when the input is an image associated to the current label.
#
# 		2) labelsArray: NumPy array which contains the labels orderer in a way
# 		that associates each label to the neuron that has generated the larger
# 		amount of spikes for it.
#
# 		3) label: current value of the label associated to the input image.

def updateClassification(spikeCountArray, labelsArray, label):
	
	# Find the most firing neuron
	mfn = mostFiringNeuron(spikeCountArray)

	# Find the neuron currently associated to the considered label
	ln = labelNeuron(labelsArray, label)

	# Switch the two elements
	labelsArray[mfn], labelsArray[ln] = labelsArray[ln], labelsArray[mfn]







# Find the most firing neuron.
#
# 	INPUT PARAMETERS: 
#
# 		spikeCountArray: NumPy array containing the count of the spikes emitted
# 		by each neuron when the input is an image associated to the current label.
#
# 	RETURN VALUES:
#
# 		The function returns the lowest index corresponding to one of the neurons
# 		that have generated the largest amount of spikes.

def mostFiringNeuron(spikeCountArray):
	return np.argmax(spikeCountArray)






# Find the neuron currently associated to the considered label.
#
# 	INPUT PARAMETERS:
#
# 		1) labelsArray: NumPy array which contains the labels orderer in a way
# 		that associates each label to the neuron that has generated the larger
# 		amount of spikes for it.
#
# 		2) label: current value of the label associated to the input image.
#
# 	RETURN VALUES:
#
# 		The function returns the index of the neuron which is currently associated
# 		to the considered label.

def labelNeuron(labelsArray, label):
	return np.where(labelsArray==label)[0][0]








# Bring the network to a quiet state in order to start a new training session with a new
# image.
#
# 	INPUT PARAMETERS:
# 		
# 		networkDictList: list of dictionaries containing the parameters of the
#		network. See the file snnDataStruct.py for more details.
#
# The function loops over all the layers and restart them. See restartLayer for more
# details.

def restartNetwork(networkDictList):

	for i in range(len(networkDictList)):

		restartLayer(networkDictList[i])






# Bring a single layer to a quiet state in order to restart the training.
#
# 	INPUT PARAMETERS:
#
# 		layerDict: dictionary associated to a specific layer. See the file
# 		snnDataStruct.py for more details.
#
# The function resets the layer as it was at the beginning of the training, leaving the
# weights array unchanged. This allows to keep the changes performed by the training and
# to prepare the network for a new training session.

def restartLayer(layerDict):
	
	layerDict["v_mem"][:] = 0
	layerDict["t_in"][:] = 0
	layerDict["t_out"][:] = 0
	layerDict["outEvents"][:] = 0





