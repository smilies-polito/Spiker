# Excitatory layer
excDict = {

	"vRest"		: -65.0,	# mV
	"vReset"	: -60.0,	# mV
	"vThresh"	: -52.0,	# mV
	"thetaPlus"	: 5, 		# mV
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
	"A_ltd"		: - 1e-3,

	# Time constants
	"ltp_tau"	: 80,		# ms
	"ltd_tau"	: 80		# ms
}

# Inter layer synapses
exc2inhWeight = 22.5			# mV
inh2excWeight = -15             	# mV


# Membrane potential time constants
tauExc = 100				# ms
tauInh = 10				# ms


# Dynamic homeostasis time constant
tauTheta = 100				# ms


