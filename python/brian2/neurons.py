import brian2 as b2
import Packages_Ontrain20606.Equations as Eqs
import Packages_Ontrain20606.Parameters as Par


def Ae(n_neurons):
    neurons = b2.NeuronGroup(
        N=n_neurons,
        model=Eqs.neuron_eqs_Ae, threshold=Eqs.thresh_Ae,
        refractory=Par.Ae_params['refac'], reset=Eqs.reset_Ae,
        namespace=Par.Ae_params,method='exact'
    )
    neurons.v = Par.Ae_params['v_reset']
    return neurons


def Ai(n_neurons):
    neurons = b2.NeuronGroup(
        N=n_neurons,
        model=Eqs.neuron_eqs_Ai, threshold=Eqs.thresh_Ai,
        refractory=Par.Ai_params['refac'], reset=Eqs.reset_Ai,
        namespace=Par.Ai_params,method='exact'
    )
    neurons.v = Par.Ai_params['v_reset']
    return neurons
