# Excitatory layer
excDict = {

	"vRest"		: -65.0,	# mV
	"vReset"	: -60.0,	# mV
	"vThresh0"	: -52.0,	# mV
	"vThreshPlus"	: 0.05,		# mV
}

# STDP synapses
stdpParam = {

	# Learning rates
	"eta_pre"	: 1e-4,		# mV
	"eta_post"	: 1e-3,		# mV

	# Time constants
	"ltp_tau"	: 20,		# ms
	"ltd_tau"	: 20		# ms
}

# Inhibitory synapses
inh2excWeight = -15	           	# mV

# Membrane potential time constant
tauExc = 100				# ms
tauThresh = 1e7				# ms

# Weights normalization factors
constSum = 100 #78.4

# Scaling factor for the random generation of the weights
scaleFactor = 0.5 #0.3

# Reference sizes used for constSum and scaleFactor
refInLayerSize = 784
refCurrLayerSize = 400
