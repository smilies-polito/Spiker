trainEquationsDict = {
	
	# Excitatory reset
	"reset_exc" 	:	'''
				v = vReset_exc
				theta += thetaPlus
				''',

	# Inhibitory reset
	"reset_inh" 	:	'v = vReset_inh',



	# Excitatory threshold
	"thresh_exc" 	:	'v >=  vThresh_exc + theta',

	# Inhibitory threshold
	"thresh_inh" 	:	'v >= vThresh_inh',




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



trainStdpDict = {


	# Stdp equations
	"stdpEqs"	:	'''
				w					: 1
				dpre/dt   =   -pre/(tc_trace)		: 1 \
					(event-driven)
				dpost/dt  = -post/(tc_trace)		: 1 \
					(event-driven)
				''',

	# Presynaptic STDP
	"stdpPre"	:	'''
				v += w
				pre = 1
				w = clip(w - nu_pre*post, 0, wMax)
				''',

	# Postsynaptic STDP
	"stdpPost"	:	'''
				post = 1
				w = clip(w + nu_post*pre, 0, wMax)
				'''

}
