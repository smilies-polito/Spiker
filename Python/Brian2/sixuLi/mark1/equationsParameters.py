import brian2 as b2

# Reset potentials
vReset_exc = -60
vReset_inh = -45.0

# Threshold voltages
vThresh_exc = -52.0
vThresh_inh = -40.0


# Time constants
tcV_exc = 100.0*b2.ms
tcV_inh = 10.0*b2.ms
tc_trace = 20*b2.ms

# Learning rates
nu_pre = 1e-4
nu_post = 1e-3


# When the neuron's membrane potential exceeds the threshold
thetaPlus = 0.1
tauTheta = 1e7*b2.ms 

# Maximum value of the weights
wMax = 1
