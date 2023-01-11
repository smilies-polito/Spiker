# Equations that characterize the model in test mode.

testEquationsDict = {
	
	# Excitatory reset
	"reset_exc" 	:	'''
				v = vReset_exc
				theta = theta + thetaPlus
				''',

	# Inhibitory reset
	"reset_inh" 	:	'v = vReset_inh',



	# Excitatory hreshold
	"thresh_exc" 	:	'v >=  vThresh_exc + theta',

	# Inhibitory threshold
	"thresh_inh" 	:	'v > vThresh_inh',




	# Excitatory equations
	"neuronsEqs_exc":	'''
				dv/dt = (vRest_exc - v)/tcV_exc	: 1
				dtheta/dt = -theta/tauTheta	: 1
				''',

	# Inhibitory equations
	"neuronsEqs_inh": 	'''
				dv/dt = (vRest_inh - v)/tcV_inh	: 1
				'''

}



testStdpDict = {


	# Stdp equations
	"stdpEqs"	:	'w : 1',

	# Presynaptic STDP
	"stdpPre"	:	'v += w',

	# Postsynaptic STDP
	"stdpPost"	:	None,
}
