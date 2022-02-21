from trainEquations import *
from testEquations import *

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
