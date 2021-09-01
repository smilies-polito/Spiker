testEquationsDict = {
	
	# Excitatory reset
	"reset_exc" 	:	'''
				v = vReset_exc
				theta = theta + thetaPlus
				timer = 0*ms
				''',

	# Inhibitory reset
	"reset_inh" 	:	'v = vReset_inh',



	# Excitatory hreshold
	"thresh_exc" 	:	'( v > (theta - offset + vThresh_exc)) and \
					(timer > tRefrac_exc)',

	# Inhibitory threshold
	"thresh_inh" 	:	'v > vThresh_inh',




	# Excitatory equations
	"neuronsEqs_exc":	'''
				dv/dt = ((vRest_exc - v) + (I_synE + \
					I_synI) /nS) / (100 * ms)	: volt
				I_synE = ge * nS *(-v)			: amp
				I_synI = gi * nS * (-100.*mV-v)		: amp
				dge/dt = -ge/(1.0*ms)			: 1
				dgi/dt = -gi/(2.0*ms)			: 1
				theta					: volt
				dtimer/dt = 0.1				: second

				''',

	# Inhibitory equations
	"neuronsEqs_inh": 	'''
				dv/dt = ((vRest_inh - v) + (I_synE + \
					I_synI) / nS) / (10 * ms)	: volt
				I_synE = ge * nS *(-v)			: amp
				I_synI = gi * nS * (-85.*mV-v)		: amp
				dge/dt = -ge/(1.0*ms)			: 1
				dgi/dt = -gi/(2.0*ms)			: 1

				'''

}



testStdpDict = {


	# Stdp equations
	"stdpEqs"	:	'w : 1',

	# Presynaptic STDP
	"stdpPre"	:	'ge = ge + w',

	# Postsynaptic STDP
	"stdpPost"	:	None,

	# Inhibitory presynaptic response when excitatory event received
	"exc2inhPre"	:	'ge = ge + w',

	# Excitatory presynaptic response when inhibitorily event reveived
	"inh2excPre"	:	'gi = gi + w'

}
