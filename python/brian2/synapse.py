import brian2 as b2
import Packages_Ontrain20606.Equations as Eqs
import Packages_Ontrain20606.Parameters as Par
import numpy as np
import torch

def X2Ae(source, target,N_source,N_target):
    synapses = b2.Synapses(
        source=source,
        target=target,
        model=Eqs.eqs_stdp_ee,
        on_pre=Eqs.eqs_stdp_pre_ee,
        on_post=Eqs.eqs_stdp_post_ee,
        namespace=Par.STDP_params
    )
    synapses.connect()
    synapses.w = 0.3 * np.reshape(torch.rand(784, 100).numpy(),(784*100))
    synapses.pre = 0
    synapses.post = 0
    return synapses

def Ae2Ai(source, target):
    synapses = b2.Synapses(
        source=source,
        target=target,
        model=Eqs.eqs_i, on_pre=Eqs.eqs_pre_i
    )
    synapses.connect("i == j")
    synapses.w = Par.exc
    return synapses

def Ai2Ae(source, target):
    synapses = b2.Synapses(
        source=source,
        target=target,
        model=Eqs.eqs_i, on_pre=Eqs.eqs_pre_i
    )
    synapses.connect("i != j")
    synapses.w = Par.inh
    return synapses