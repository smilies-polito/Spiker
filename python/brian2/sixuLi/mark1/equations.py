from trainEquations import *
from testEquations import *


excInhConnectionDict = {

	"exc2inhEqs"	:	'w	: 1',

	"inh2excEqs"	:	'w	: 1',



	# Inhibitory presynaptic response when excitatory event received
	"exc2inhPre"	:	'v_post += w',

	# Excitatory presynaptic response when inhibitorily event reveived
	"inh2excPre"	:	'v_post += w'

}

def defineEquations(mode):

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
