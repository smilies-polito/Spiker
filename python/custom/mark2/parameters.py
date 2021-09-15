# Excitatory layer
excDict = {

	"vRest"		: -65.0,	# mV
	"vReset"	: -60.0,	# mV
	"vThresh"	: -52.0,	# mV
	"thetaPlus"	: 0.01,		# mV
	"initTheta"	: 20		# mV

}


# Inhibitory layer
inhDict = {

	"vRest"		: -60.0,	# mV
	"vReset"	: -45.0,	# mV
	"vThresh"	: -40.0		# mV

}


# STDP synapses
stdpDict = {

	# Learning rates
	"A_ltp"		: 1e-4,
	"A_ltd"		: 1e-3,

	# Time constants
	"ltp_tau"	: 20,		# ms
	"ltd_tau"	: 20		# ms
}

# Inter layer synapses
exc2inhWeight = 21			# mV
inh2excWeight = -15	           	# mV


# Membrane potential time constants
tauExc = 100				# ms
tauInh = 10				# ms


# Dynamic homeostasis time constant
tauTheta = 1e5				# ms

# Weights normalization factor
constSum = 78.4

# Scaling factor for the random generation of the weights
scaleFactor = 0.3	
