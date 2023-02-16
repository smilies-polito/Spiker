from trainEquations import *
from testEquations import *

def defineEquations(mode):

	"""
	Define the network equations based on the execution mode.

	INPUT:
		mode: string. Can be "train" or "test"

	OUTPUT:
		Dictionaries containing the equation for the neuron model and
		the stdp
		
	"""

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
