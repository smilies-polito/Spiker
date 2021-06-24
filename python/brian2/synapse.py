import brian2 as b2
import equations as Eqs
import parameters as Par
import numpy as np
import torch

def excToExcConnection(source, target, N_source,N_target):
    synapses = b2.Synapses(
        source=source,
        target=target,
        model=Eqs.eqs_stdp_ee,
        on_pre=Eqs.eqs_stdp_pre_ee,
        on_post=Eqs.eqs_stdp_post_ee,
        namespace=Par.STDP_params
    )
    synapses.connect()
    synapses.w = 0.3 * np.reshape(torch.rand(N_source,
    N_target).numpy(),(N_source*N_target))
    synapses.pre = 0
    synapses.post = 0
    return synapses


def excToInhConnection(source, target):
    synapses = b2.Synapses(
        source=source,
        target=target,
        model=Eqs.eqs_i, on_pre=Eqs.eqs_pre_i
    )
    synapses.connect("i == j")
    synapses.w = Par.exc
    return synapses

def inhToExcConnection(source, target):
    synapses = b2.Synapses(
        source=source,
        target=target,
        model=Eqs.eqs_i, on_pre=Eqs.eqs_pre_i
    )
    synapses.connect("i != j")
    synapses.w = Par.inh
    return synapses
