from bitWidths import *
from utils import fixedPoint

# Excitatory layer
excDict = {			# Shifted values to minimize the interval
				# ----------------------------------------------
				# |Original|	|  Ref  |	|Shift |
				# |--------|----|-------|-------|------|
	"vRest"		: 0,	# | -65.0  | =	| -65.0 | + 	| 0.0  | mV
	"vReset"	: 0,	# | -60.0  | =	| -65.0 | + 	| 5.0  | mV
	"vThresh0"	: 1,	# | -52.0  | =	| -65.0 | + 	| 13.0 | mV
	"vThreshPlus"	: 0.05,	# |  0.05  | =	|   -   | + 	|  -   | mV
}


# Finite precision
excDict["vRest"] = fixedPoint(excDict["vRest"], fixed_point_decimals,
		neuron_bitWidth)
excDict["vReset"] = fixedPoint(excDict["vReset"], fixed_point_decimals,
		neuron_bitWidth)
excDict["vThresh0"] = fixedPoint(excDict["vThresh0"], fixed_point_decimals,
		neuron_bitWidth)
excDict["vThreshPlus"] = fixedPoint(excDict["vThreshPlus"],
				fixed_point_decimals, neuron_bitWidth)

# Membrane potential time constant
tauExc = 100				# ms
