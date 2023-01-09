reset_Ae = '''
v = v_reset
theta = theta + theta_plus
'''

thresh_Ae = 'v >= (v_thresh + theta)'

neuron_eqs_Ae = '''
dv/dt = ((v_rest - v) ) / tc_v  : 1
theta                           : 1
'''

reset_Ai = 'v = v_reset'
thresh_Ai = 'v >= v_thresh'

neuron_eqs_Ai = '''
dv/dt = ((v_rest - v) ) / tc_v  : 1
'''

eqs_stdp_ee = '''
w                             : 1
dpre/dt = -pre / tc_trace     : 1 (event-driven)
dpost/dt = -post / tc_trace   : 1 (event-driven)
'''

eqs_stdp_pre_ee = '''
pre = 1
v_post += w
'''

eqs_stdp_post_ee = '''
post = 1
w = w + nu_post * pre
'''

eqs_i = '''
w                             : 1
'''

eqs_pre_i = '''
v_post += w
'''