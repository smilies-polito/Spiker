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


def netCreate():
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

		initialValues = {
			"rates": np.random.randn(784)*b2.Hz
		}
		poissonGroup.set_states(initialValues)

		weightMatrix = (b2.random(networkList[0]*networkList[1]) + 0.01)*0.3

# Stdp connection between Poisson layer and first excitatory layer
		exc2exc = b2.Synapses(poissonGroup, excNeurons, 
		model = stdpEqs, on_pre = stdpPre, on_post = stdpPost, method = 'exact')
		exc2exc.connect(None)

# Initialize the weights
		exc2exc.w = weightMatrix

# Create the monitor for the output spikes
		spikeMonitor = b2.SpikeMonitor(excNeurons, record=False)



		net = b2.Network(excNeurons, inhNeurons, exc2exc, inh2exc, exc2inh, poissonGroup)
		net.add(spikeMonitor)

		return net


net = netCreate()
net.run(10*b2.ms)

for i in net.get_states(units=True, format='dict', subexpressions=False,
	read_only_variables=True, level=0):

	print(i)

#print(net.get_states(units=True, format='dict', subexpressions=False,
#	read_only_variables=True, level=0))

values = {
	"poissongroup":{
		"rates": np.ones(784)*b2.Hz
	}
}

net.set_states(values, units=True, format='dict', level=0)

print(net.get_states(units=True, format='dict', subexpressions=False,
	read_only_variables=True, level=0)["poissongroup"]["rates"])


