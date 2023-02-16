import brian2 as b2

# Reset potentials
vReset_exc	= -60		# mV
vReset_inh	= -45.0		# mV

# Threshold voltages
vThresh_exc	= -52.0		# mV
vThresh_inh	= -40.0		# mV


# Time constants
tcV_exc		= 100.0*b2.ms
tcV_inh		= 10.0*b2.ms
tc_trace	= 20*b2.ms

# Learning rates
nu_pre		= 1e-4		# mV
nu_post		= 1e-3		# mV


# When the neuron's membrane potential exceeds the threshold
thetaPlus	= 0.1		# mV
tauTheta	= 1e7*b2.ms 

# Maximum value of the weights
wMax 		= 2**16		# mV
