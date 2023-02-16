# Excitatory layer
excDict = {

	"vRest"		: -65.0,	# mV
	"vReset"	: -60.0,	# mV
	"vThresh0"	: -52.0,	# mV
	"vThreshPlus"	: 5,		# mV
	"initThreshold"	: -40		# mV

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


# Dynamic homeostasis time constant
tauThresh = 1e2				# ms

# Weights normalization factor
constSum = 78.4

# Scaling factor for the random generation of the weights
scaleFactor = 0.3	
