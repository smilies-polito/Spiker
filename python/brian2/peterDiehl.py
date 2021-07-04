import brian2 as b2
import timeit
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold = np.inf, linewidth = 150)


from neurons import *
from synapse import *
from trainFunctions import *

# To graphically visualize the synapse connections between layers
from plotUtils import *

from mnist import loadDataset


# Script that implements the model presented in "Unsupervised learning of digit
# recognition using stdp"



# PARAMETERS
# -----------------------------------------------------------------------------

# MNIST files
images = "./mnist/train-images-idx3-ubyte"
labels = "./mnist/train-labels-idx1-ubyte"

# Layers of the network
networkList = [784, 400]

# Train periods
singleExampleTime = 0.35*b2.second
restingTime = 0.15*b2.second

updateInterval = 250
weightUpdateInterval = 10
printInterval = 1
startInputIntensity = 2.
inputIntensity = startInputIntensity

# Rest potentials
vRest_exc = -65.*b2.mV
vRest_inh = -60.*b2.mV

# Reset potentials
vReset_exc = -65.*b2.mV
vReset_inh =  -45.*b2.mV

# Threshold voltages
vThresh_exc = -52*b2.mV
vThresh_inh = -40.*b2.mV

# Refractory periods
tRefrac_exc = 5.*b2.ms
tRefrac_inh = 2.*b2.ms

# Constant total sum of the weights for normalization
totalWeight = 78

delay_in2exc = 10*b2.ms
delay_exc2inh = 5*b2.ms



# Time constants
tauPre_exc = 20*b2.ms
tauPost1_exc = 20*b2.ms
tauPost2_exc = 40*b2.ms

# Learning rates
etaPre_exc = 0.0001
etaPost_exc = 0.01
etaPre_AeAe = 0.1
etaPost_AeAe = 0.5

# Weight dependence
wMax_exc = 1.0
nuPre_exc = 0.2
nuPost_exc = 0.2
wMu_pre = 0.2
wMu_post = 0.2



# EQUATIONS
# -----------------------------------------------------------------------------

# When the neuron's membrane potential exceeds the threshold
tauTheta = 1e7*b2.ms
thetaPlus = 0.05*b2.mV
reset_exc = '''
	v = vReset_exc
	theta = theta + thetaPlus
	timer = 0*ms
'''
reset_inh = 'v = vReset_inh'


offset = 20.0*b2.mV

# Thresholds
thresh_exc = '( v > (theta - offset + vThresh_exc)) and (timer > \
tRefrac_exc)'
thresh_inh = 'v > vThresh_inh'

# Resets


# Equations for the neurons
neuronsEqs_exc = '''
	dv/dt = ((vRest_exc - v) + (I_synE + I_synI) / nS) / (100 * ms)	: volt
        I_synE = ge * nS *(-v)						: amp
        I_synI = gi * nS * (-100.*mV-v)					: amp
        dge/dt = -ge/(1.0*ms)						: 1
        dgi/dt = -gi/(2.0*ms)						: 1
	dtheta/dt = -theta/tauTheta					: volt
	dtimer/dt = 0.1							: second

'''

neuronsEqs_inh = '''
	dv/dt = ((vRest_inh - v) + (I_synE + I_synI) / nS) / (10 * ms)	: volt
        I_synE = ge * nS *(-v)						: amp
        I_synI = gi * nS * (-85.*mV-v)					: amp
        dge/dt = -ge/(1.0*ms)						: 1
        dgi/dt = -gi/(2.0*ms)						: 1

'''


# Stdp equations
stdpEqs = '''
	w					: 1
	post2before				: 1
	dpre/dt   =   -pre/(tauPre_exc)		: 1	(event-driven)
	dpost1/dt  = -post1/(tauPost1_exc)	: 1	(event-driven)
	dpost2/dt  = -post2/(tauPost2_exc)	: 1	(event-driven)	

'''

stdpPre = '''
	ge = ge + w
	pre = 1.
	w = clip(w - etaPre_exc*post1, 0, wMax_exc)
'''

stdpPost = '''
	post2before = post2
	w = clip(w + etaPost_exc * pre * post2before, 0, wMax_exc)
	post1 = 1.
	post2 = 1.	
'''




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

# Create the monitor for the output spikes
spikeMonitor = b2.SpikeMonitor(excNeurons)
membraneMonitor = b2.StateMonitor(excNeurons, 'v', record = True)

# Assignements of the neurons to the labels
assignements = -1 * np.ones(networkList[1])



i = 0
b2.run(0*b2.ms)
prevSpikesCount = 0

startTimeTraining = timeit.default_timer()

accuracies = []

while i < imgArray.shape[0]:

	# Convert the image into a Poisson train of spikes
	poissonGroup.rates = imgArray[i]/8*inputIntensity * b2.Hz

	# Normalize the weights
	#exc2exc.w = exc2exc.w/np.sum(exc2exc.w)*totalWeight

	# Measure the time at which the training on the single image starts
	startTimeImage = timeit.default_timer()

	# Train the network
	b2.run(singleExampleTime, profile = True)
	b2.profiling_summary(show=5)

	printString = "Time to run: " + str(timeit.default_timer() -
	startTimeImage)
	print(printString)

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
