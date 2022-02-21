# Excitatory layer
excDict = {

	"vRest"		: -65.0,	# mV
	"vReset"	: -60.0,	# mV
	"vThresh0"	: -52.0,	# mV
	"vThreshPlus"	: 0.05,		# mV
}

# STDP synapses
stdpDict = {

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
constSum0 = 78.4
constSum1 = 78.4 * (784/200) * (400/100)

# Scaling factor for the random generation of the weights
scaleFactor0 = 0.3
scaleFactor1 = 0.3 * (784/200) * (400/100)
