# Excitatory layer
excDict = {

	"vRest"		: -65.0,	# mV
	"vReset"	: -60.0,	# mV
	"vThresh"	: -52.0,	# mV
	"thetaPlus"	: 0.1,		# mV
	"initTheta"	: 20		# mV

}


# Inhibitory layer
inhDict = {

	"vRest"		: -60.0,	# mV
	"vReset"	: -45.0,	# mV
	"vThresh"	: -40.0		# mV
}


# Inter layer synapses
exc2inhWeight = 22.5		# mV
inh2excWeight = -15             # mV


# Membrane potential time constants
tauExc = 100			# ms
tauInh = 10			# ms


# STDP time constant
tauTraces = 20			# ms


# Dynamic homeostasis time constant
tauTheta = 1e7			# ms

# Learning rates
A_ltp = 1e-4
A_ltd = 1e-3
