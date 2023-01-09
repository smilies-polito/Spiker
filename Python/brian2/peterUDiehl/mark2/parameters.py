import brian2 as b2


# Reset potentials
vReset_exc = -65.*b2.mV
vReset_inh =  -45.*b2.mV

# Threshold voltages
vThresh_exc = -52*b2.mV
vThresh_inh = -40.*b2.mV

# Constant total sum of the weights for normalization
constSum = 78

# Scaling factor for the random generation of the weights
scaleFactor = 0.3

# Reference size used for constSum and scaleFactor
refInLayerSize = 784
refCurrLayerSize = 400

delay_inh2exc = 10*b2.ms
delay_exc2inh = 5*b2.ms

# Time constants
tau_pre_exc = 20*b2.ms
tau_post1_exc = 20*b2.ms
tau_post2_exc = 40*b2.ms

# Learning rates
eta_pre_exc = 0.0001
eta_post_exc = 0.01
eta_pre_exc2exc = 0.1
eta_post_exc2exc = 0.5

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


parametersDict = {

	# Excitatory rest potential
	"vRest_exc"	: -65.*b2.mV,

	# Inhibitory rest potential
	"vRest_inh"	: -60.*b2.mV,


	# Excitatory refractory period
	"tRefrac_exc"	: 5.*b2.ms,

	# Inhibitory refractory period
	"tRefrac_inh"	: 2.*b2.ms

}

weightInitDict = {
	"exc2inh"	: 10.4,
	"inh2exc"	: 17.4
}
