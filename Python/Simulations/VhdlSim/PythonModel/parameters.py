from bitWidths import *
from utils import fixedPoint

# Excitatory layer
excDict = {			# Shifted values to minimize the interval
				# ----------------------------------------------
				# |Original|	|  Ref  |	|Shift |
				# |--------|----|-------|-------|------|
	"vRest"		: 0,	# | -65.0  | =	| -65.0 | + 	| 0.0  | mV
	"vReset"	: 5,	# | -60.0  | =	| -65.0 | + 	| 5.0  | mV
	"vThresh0"	: 13,	# | -52.0  | =	| -65.0 | + 	| 13.0 | mV
	"vThreshPlus"	: 0.05,	# |  0.05  | =	|   -   | + 	|  -   | mV
}


# Finite precision
excDict["vRest"] = fixedPoint(excDict["vRest"], fixed_point_decimals)
excDict["vReset"] = fixedPoint(excDict["vReset"], fixed_point_decimals)
excDict["vThresh0"] = fixedPoint(excDict["vThresh0"], fixed_point_decimals)
excDict["vThreshPlus"] = fixedPoint(excDict["vThreshPlus"],
				fixed_point_decimals)


# STDP synapses
stdpDict = {

	# Learning rates
	"eta_pre"	: 1e-4,		# mV
	"eta_post"	: 1e-3,		# mV

	# Time constants
	"ltp_tau"	: 20,		# ms
	"ltd_tau"	: 20		# ms
}

# Finite precision
stdpDict["eta_pre"] = fixedPoint(stdpDict["eta_pre"], fixed_point_decimals) 
stdpDict["eta_post"] = fixedPoint(stdpDict["eta_post"], fixed_point_decimals) 
stdpDict["ltp_tau"] = fixedPoint(stdpDict["ltp_tau"], fixed_point_decimals) 
stdpDict["ltd_tau"] = fixedPoint(stdpDict["ltd_tau"], fixed_point_decimals) 

# Inhibitory synapses
inh2excWeight = -15	           	# mV

# Finite precision
inh2excWeight = fixedPoint(inh2excWeight,  fixed_point_decimals)

# Membrane potential time constant
tauExc = 100				# ms

# Weights normalization factor
constSum = 78.4

constSum = fixedPoint(constSum,  fixed_point_decimals)

# Scaling factor for the random generation of the weights
scaleFactor = 0.3	
