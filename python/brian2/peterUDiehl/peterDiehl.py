import brian2 as b2
import timeit
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
import sys

np.set_printoptions(threshold = np.inf, linewidth = 150)

from trainFunctions import *

# To graphically visualize the synapse connections between layers
from plotUtils import *

from mnist import loadDataset

from equations import *
from parameters import *


# Script that implements the model presented in "Unsupervised learning of digit
# recognition using stdp"



# PARAMETERS
# -----------------------------------------------------------------------------

# MNIST files
images = "../mnist/train-images-idx3-ubyte"
labels = "../mnist/train-labels-idx1-ubyte"

# Layers of the network
networkList = [784, 400]


updateInterval = 250
printInterval = 10
startInputIntensity = 2.
inputIntensity = startInputIntensity



# NEURON GROUPS
# -----------------------------------------------------------------------------

# Excitatory layer
excNeurons = b2.NeuronGroup(networkList[1], neuronsEqs_exc, threshold =
thresh_exc, refractory = tRefrac_exc, reset = reset_exc, method = 'euler')

# Inhibitory layer
inhNeurons = b2.NeuronGroup(networkList[1], neuronsEqs_inh, threshold =
thresh_inh, refractory = tRefrac_inh, reset = reset_inh, method = 'euler')

# Initialize the membrane potential
excNeurons.v = vRest_exc - 40.*b2.mV
inhNeurons.v = vRest_inh - 40.*b2.mV

# Initialize the threshold parameter theta
excNeurons.theta = np.ones(networkList[1])*20.0*b2.mV






# SYNAPSES
# -----------------------------------------------------------------------------

# Excitatory to inhibitory one to one connection
exc2inh = b2.Synapses(excNeurons, inhNeurons, 'w : 1', 
on_pre = 'ge = ge + w')
exc2inh.connect('i==j')
exc2inh.w = 10.4

# Inhibitory to excitatory connection
inh2exc = b2.Synapses(inhNeurons, excNeurons, 'w : 1', 
on_pre = 'gi = gi + w')
inh2exc.connect('i!=j')
inh2exc.w = 17.4





# INPUT NEURONS GROUP AND STDP CONNECTION
# -----------------------------------------------------------------------------

poissonGroup = b2.PoissonGroup(networkList[0], 0*b2.Hz)
weightMatrix = (b2.random(networkList[0]*networkList[1]) + 0.01)*0.3

# Stdp connection between Poisson layer and first excitatory layer
exc2exc = b2.Synapses(poissonGroup, excNeurons, 
model = stdpEqs, on_pre = stdpPre, on_post = stdpPost, method = 'exact')
exc2exc.connect()

# Initialize the weights
exc2exc.w = weightMatrix





# LOAD THE MNIST DATASET
# -----------------------------------------------------------------------------
imgArray, labelsArray = loadDataset(images, labels)





# RUN THE TRAINING
# -----------------------------------------------------------------------------

# Allocate the array that monitors the evolution of the number of spikes
# generated by each neuron for an "updateInterval" number of images
spikesEvolution = np.zeros((updateInterval, networkList[1]))


# Assignements of the neurons to the labels
assignements = -1 * np.ones(networkList[1])

# Create the monitor for the output spikes
spikeMonitor = b2.SpikeMonitor(excNeurons, record=False)



i = 0
prevSpikesCount = 0

startTimeTraining = timeit.default_timer()

accuracies = []

tracemalloc.start()


local_vars = list(locals().items())


while i < imgArray.shape[0]:


	# Convert the image into a Poisson train of spikes
	poissonGroup.rates = imgArray[i]/8*inputIntensity * b2.Hz

	# Normalize the weights
	#exc2exc.w = exc2exc.w/np.sum(exc2exc.w)*totalWeight

	# Measure the time at which the training on the single image starts
	startTimeImage = timeit.default_timer()

	# Train the network
	b2.run(singleExampleTime)

	# Store the count of the spikes for the current image
	currentSpikesCount = spikeMonitor.count - prevSpikesCount
	prevSpikesCount = np.asarray(spikeMonitor.count)


	if np.sum(currentSpikesCount) < 5:

		inputIntensity +=1
		print("Increase input intensity: ", inputIntensity)

	else:
		spikesEvolution[i % updateInterval] = currentSpikesCount

		printProgress(i, printInterval, startTimeImage, startTimeTraining)

		accuracies = computePerformances(i, updateInterval, networkList[1],
			spikesEvolution, labelsArray[i - updateInterval : i], 
			assignements, accuracies)

		# Update the correspondence between the output neurons and the labels
		updateAssignements(i, updateInterval, networkList[1], spikesEvolution,
			labelsArray[i - updateInterval : i], assignements)

		inputIntensity = startInputIntensity

		i += 1


	poissonGroup.rates[:] = 0
	b2.run(restingTime)

	print(tracemalloc.get_traced_memory())

tracemalloc.stop()


fp = open("accuracies.txt", "w")
fp.write(str(accuracies))
fp.close()
