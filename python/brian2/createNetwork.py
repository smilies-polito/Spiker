import brian2 as b2
from neurons import *
from synapse import *

# Script that implements the model presented in "Unsupervised learning of digit
# recognition using stdp"

# To graphically visualize the synapse connections between layers
from plotUtils import *


# PARAMETERS
# -----------------------------------------------------------------------------

# Layers of the network
networkList = [784, 400]

# Train periods
singleExampleTime = 0.35*b2.second
restingTime = 0.15*b2.second

updateInterval = 250
weightUpdateInterval = 10
printInterval = 10

# Rest potentials
vRest_exc = -65.*b2.mV
vRest_inh = -60.*b2.mV

# Reset potentials
vReset_exc = -65.*b2.mV
vReset_inh = -45.*b2.mV

# Threshold voltages
vThresh_exc = -52.*b2.mV
vThresh_inh = -40.*b2.mV

# Refractory periods
tRefrac_exc = 5.*b2.ms
tRefrac_inh = 2.*b2.ms

# Constant total sum of the weights for normalization
totalWeight = 78

delay_in2exc = 10*b2.ms
delay_exc2inh = 5*b2.ms

inputIntensity = 2.

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

offset = 20.0*b2.mV
vThresh_exc = '( v > (theta - offset + ' + str(vThresh_exc) + ')) * (timer > \
tRefrac_exc)'

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
	w					: siemens
	post2before				: 1
	dpre/dt   =   -pre/(tc_pre_ee)		: 1	(clock-driven)
	dpost1/dt  = -post1/(tc_post_1_ee)	: 1	(clock-driven)
	dpost2/dt  = -post2/(tc_post_2_ee)	: 1	(clock-driven)	

'''

stdpPre = '''
	ge = ge + w
	pre = 1.
	w = clip(w - etaPre_exc*post1, 0, wMax)
'''

stdpPost = '''
	post2before = post2
	w = clip(w + etaPost_exc * pre * post2before, 0, wMax)
	post1 = 1.
	post2 = 1.	
'''








# NEURON GROUPS
# -----------------------------------------------------------------------------

# Excitatory layer
excNeurons = b2.NeuronGroup(networkList[1], neuronsEqs_exc, threshold =
vThresh_exc, refractory = tRefrac_exc, reset = reset_exc)

# Inhibitory layer
inhNeurons = b2.NeuronGroup(networkList[1], neuronsEqs_inh, threshold =
vThresh_inh, refractory = tRefrac_inh, reset = vReset_inh)

# Initialize the membrane potential
excNeurons.v = vRest_exc - 40.*b2.mV
inhNeurons.v = vRest_inh - 40.*b2.mV

# Initialize the threshold parameter theta
excNeurons.theta = np.ones(networkList[1])*20.0*b2.mV









# SYNAPSES
# -----------------------------------------------------------------------------

# Excitatory to inhibitory one to one connection
exc2inh = b2.Synapses(excNeurons, inhNeurons, 'w : siemens', 
on_pre = 'ge = ge + w')
exc2inh.connect('i==j')
exc2inh.w = 10.4*b2.mS

# Inhibitory to excitatory connection
inh2exc = b2.Synapses(inhNeurons, excNeurons, 'w : siemens', 
on_pre = 'gi = gi + w')
inh2exc.connect('i!=j')
inh2exc.w = 17.4*b2.mS








# INPUT NEURONS GROUP AND STDP CONNECTION
# -----------------------------------------------------------------------------

poissonGroup = b2.PoissonGroup(networkList[0], 0*b2.Hz)
weightMatrix = (b2.random(networkList[0]*networkList[1]) + 0.01)*0.3
exc2exc = b2.Synapses(poissonGroup, excNeurons, 
model = stdpEqs, on_pre = stdpPre, on_post = stdpPost)
exc2exc.connect()
exc2exc.w = weightMatrix*b2.mS
