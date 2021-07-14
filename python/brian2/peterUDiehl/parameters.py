import brian2 as b2


# Train periods
singleExampleTime = 0.35*b2.second
restingTime = 0.15*b2.second

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

# When the neuron's membrane potential exceeds the threshold
tauTheta = 1e7*b2.ms
thetaPlus = 0.05*b2.mV

offset = 20.0*b2.mV
