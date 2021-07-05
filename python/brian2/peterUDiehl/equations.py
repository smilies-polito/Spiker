# Resets
reset_exc = '''
	v = vReset_exc
	theta = theta + thetaPlus
	timer = 0*ms
'''
reset_inh = 'v = vReset_inh'



# Thresholds
thresh_exc = '( v > (theta - offset + vThresh_exc)) and (timer > \
tRefrac_exc)'
thresh_inh = 'v > vThresh_inh'




# Equations for the neurons
neuronsEqs_exc = '''
	dv/dt = ((vRest_exc - v) + (I_synE + I_synI) / nS) / (100 * ms)	: volt
        I_synE = ge * nS *(-v)						: amp
        I_synI = gi * nS * (-100.*mV-v)					: amp
        dge/dt = -ge/(1.0*ms)						: 1
        dgi/dt = -gi/(2.0*ms)						: 1
	dtheta/dt = -theta/tauTheta					: volt
	dtimer/dt = 0.1							: second

'''

neuronsEqs_inh = '''
	dv/dt = ((vRest_inh - v) + (I_synE + I_synI) / nS) / (10 * ms)	: volt
        I_synE = ge * nS *(-v)						: amp
        I_synI = gi * nS * (-85.*mV-v)					: amp
        dge/dt = -ge/(1.0*ms)						: 1
        dgi/dt = -gi/(2.0*ms)						: 1

'''


# Stdp equations
stdpEqs = '''
	w					: 1
	post2before				: 1
	dpre/dt   =   -pre/(tauPre_exc)		: 1	(event-driven)
	dpost1/dt  = -post1/(tauPost1_exc)	: 1	(event-driven)
	dpost2/dt  = -post2/(tauPost2_exc)	: 1	(event-driven)	

'''

stdpPre = '''
	ge = ge + w
	pre = 1.
	w = clip(w - etaPre_exc*post1, 0, wMax_exc)
'''

stdpPost = '''
	post2before = post2
	w = clip(w + etaPost_exc * pre * post2before, 0, wMax_exc)
	post1 = 1.
	post2 = 1.	
'''
