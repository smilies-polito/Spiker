import brian2 as b2
Ae_params = {
    'v_thresh': -52.0,
    'v_reset': -60.0,
    'v_rest': -65.0,
    'tc_v': 100.0 * b2.ms,
    'theta_plus': 0.05,
    'refac': 5 * b2.ms
}

Ai_params = {
    'v_thresh': -40.0,
    'v_reset': -45.0,
    'v_rest': -60.0,
    'tc_v': 10.0 * b2.ms,
    'refac': 2 * b2.ms
}

STDP_params = {
    'nu_pre': 1e-4,
    'nu_post': 1e-2,
    'tc_trace':20 * b2.ms
}

exc = 22.5
inh = -120
# inh = -60