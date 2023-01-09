from trainEquations import *
from testEquations import *


excInhConnectionDict = {

	"exc2inhEqs"	:	'w	: 1',

	"inh2excEqs"	:	'w	: 1',

	# Inhibitory presynaptic response when excitatory event received
	"exc2inhPre"	:	'v += w',

	# Excitatory presynaptic response when inhibitorily event reveived
	"inh2excPre"	:	'v += w'

}


def defineEquations(mode):

	'''
	Select the correct equations depending on the operational mode.

	INPUT:

		mode: string. It can be "train" or "test".

	OUTPUT:

		touple of dictionaries containing the membrane potential and the
		STDP equations.

	'''

	if mode == "train":
		return trainEquationsDict, trainStdpDict

	elif mode == "test":
		return testEquationsDict, testStdpDict

	else:
		print('''
			Invalid operation mode. Accepted values:
				1) test
				2) train
		''')
