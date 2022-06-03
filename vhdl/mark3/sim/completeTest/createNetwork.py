import numpy as np
import sys
from utils import fixedPoint, fixedPointArray
from storeParameters import storeArray


def createNetwork(networkList, weightFilename, thresholdFilename, mode,
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, trainPrecision, rng):

	'''
	Create the complete network dictionary.

	INPUT:

		1) networkList: list of integer numbers. Each element of the
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) weightFilename: string. Root of the weights file name of each
		layer. The function appends the number of the current layer to
		it.

		3) threhsoldFilename: string. Root of the threhsholds file name
		of each layer. The function appends the number of the current
		layer to it.

		4) mode: string. It can be "train" or "test".

		5) excDictList: list of dictionaries, each containing the
		initialization values for a specific excitatory layer.

		6) scaleFactors: float NumPy array. Factor used to scale the
		randomly generated weights for each layer. Needed in training
		mode. In test mode "None" can be used.

		7) inh2excWeights: float NumPy array. Weight of the synapse for
		each layer. This is the same for all the connections within the
		layer.

		9) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

		9) trainPrecision: string. Numerical precision of the training.
		It can be "fixedPoint" or "float". Needed only in test mode. 

		10) rng: NumPy random generator.
	'''

	network = {}

	for layer in range(1, len(networkList)):

		if mode == "test":
			weightFile = weightFilename + str(layer) + ".npy"
			thresholdFile = thresholdFilename + str(layer) + ".npy"
		else:
			weightFile = None
			thresholdFile = None

		# Create the excitatory layer
		createLayer(network, "exc", excDictList[layer-1], networkList,
				layer, mode, thresholdFile,
				fixed_point_decimals, trainPrecision)

		# Create the excitatory to excitatory connection
		intraLayersSynapses(network, "exc2exc", mode, networkList,
				weightFile, layer, scaleFactors[layer-1],
				fixed_point_decimals, trainPrecision, rng)

		# Create the inhibitory to excitatory connection
		interLayerSynapses(network, "inh2exc", inh2excWeights[layer-1],
					layer)

	return network






def createLayer(network, layerType, initDict, networkList, layer, mode,
		thresholdFile, fixed_point_decimals, trainPrecision):

	'''
	Create the layer dictionary and add it to the network dictionary.

	INPUT:
		1) network: dictionary of the network.

		2) layerType: string. It can be "inh" or "exc".

		3) initDict: dictionary containing the initialization values for
		the the layer.

		4) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		5) layer: index of the current layer. The count starts from 1.

		6) mode: string. It can be "train" or "test".

		7) thresholdFile: complete name of the file containing the
		thresholds for the current layers.
	
		8) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

		9) trainPrecision: string. Numerical precision of the training.
		It can be "fixedPoint" or "float". Needed only in test mode. 
	'''

	# Create the name for the layer
	layerName = layerType + "Layer" + str(layer)

	network[layerName] = {

		# Initialize the membrane potentials at the reset voltage		
		"v"		: initDict["vRest"]*np.ones((1, 
					networkList[layer])).astype(int),

		# Initialize the threshold potential
		"vThresh0"	: initDict["vThresh0"],

		# Initialize the rest potential
		"vRest"		: initDict["vRest"],

		# Initialize the reset potential
		"vReset"	: initDict["vReset"],

		# Initialize the homeostasis parameter
		"vThreshPlus"	: initDict["vThreshPlus"],

		# Initialize the dynamic homeostasis
		"vThresh" 	: initializeThreshold(mode, thresholdFile,
						initDict, networkList[layer],
						fixed_point_decimals,
						trainPrecision),

		# Initialize the output spikes
		"outSpikes"	: np.zeros((1,
			networkList[layer])).astype(bool),

		# Initialize the output spikes
		"inhSpikes"	: np.zeros((1,
			networkList[layer])).astype(bool)
	}






def initializeThreshold(mode, thresholdFile, initDict, numberOfNeurons,
		fixed_point_decimals, trainPrecision):

	'''
	Initialize the dynamic homeostasis parameter theta.

	INPUT:

		1) mode: string. It can be "train" or "test".

		2) thresholdFile: complete name of the file containing the
		thresholds for the current layers.

		3) initDict: dictionary containing the initialization values for
		the the layer.

		4) numberOfNeurons: number of neurons in the layer.

		5) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

		6) trainPrecision: string. Numerical precision of the training.
		It can be "fixedPoint" or "float". Needed only in test mode. 

	The function initialize the theta parameter depending on the mode in
	which the network will be run, train or test.

	'''

	if mode == "train":

		# Initialize the thresholds to a starting value
		return initDict["vThresh0"]*np.ones((1,
			numberOfNeurons)).astype(int)

	elif mode == "test":

		# Load thresholds values from file
		with open(thresholdFile, 'rb') as fp: 
			thresholds = np.load(fp)

		# Check if the thresholds are already in fixed point or not
		if trainPrecision == "fixedPoint":
			return thresholds

		elif trainPrecision == "float":
			return fixedPointArray(thresholds, fixed_point_decimals)

		else:
			# Invalid mode, print error and exit
			print('Invalid  training precision. Accepted values:\
				\n\t1) fixedPoint \n\t2) float')
			sys.exit()

	else:

		# Invalid mode, print error and exit
		print('Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train')
		sys.exit()





def intraLayersSynapses(network, synapseName, mode, networkList, weightFile,
			layer, scaleFactor, fixed_point_decimals,
			trainPrecision, rng):

	'''	
	Initialize the intra layer synapses and add it to the network dictionary.

	INPUT:

		1) network: dictionary of the network.

		2) synapseName: string reporting the name of the connection. The
		standard name is "exc2exc". The function appends the number of
		the current layer.

		3) mode: string. It can be "train" or "test".
		
		4) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		5) weightFile: Complete name of the file containing the trained
		weights for the current layer. Needed in test mode. In training
		mode "None" can be used.

		6) layer: index of the current layer. The count starts from 1.
		Needed in training mode. In test mode "None" can be used.

		7) scaleFactor: float number. Factor used to scale the randomly
		generated weights. Needed in training mode. In test mode "None" 
		can be used.
		
		8) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

		9) trainPrecision: string. Numerical precision of the training.
		It can be "fixedPoint" or "float". Needed only in test mode. 

		10) rng: NumPy random generator.
	'''
	
	# Append the number of the current layer to the name
	synapseName = synapseName + str(layer)

	network[synapseName] = {

		# Initialize the synapses weights
		"weights"	: initializeWeights(mode, networkList,
					weightFile, layer, scaleFactor,
					fixed_point_decimals, trainPrecision,
					rng),

		# Initialize the pre-synaptic trace
		"pre"		: np.zeros((1, networkList[layer - 1])),

		# Initialize the post-synaptic trace
		"post"		: np.zeros((networkList[layer], 1))	

	}







def initializeWeights(mode, networkList, weightFile, layer, scaleFactor,
		fixed_point_decimals, trainPrecision, rng):

	'''
	Initialize the weights of the connections between two layers.

	INPUT:	
		1) mode: string. It can be "train" or "test".
		
		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) weightFile: Complete name of the file containing the trained
		weights for the current layer. Needed in test mode. In training
		mode "None" can be used.

		4) layer: index of the current layer. The count starts from 1.
		Needed in training mode. In test mode "None" can be used.

		5) scaleFactor: float number. Factor used to scale the randomly
		generated weights. Needed in training mode. In test mode "None" 
		can be used.

		6) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

		7) trainPrecision: string. Numerical precision of the training.
		It can be "fixedPoint" or "float". Needed only in test mode. 

		8) rng: NumPy random generator.

	The function initializes the weights depending on the mode in
	which the network will be run, train or test.

	'''

	if mode == "train":

		# Randomly initialize the weights
		weights = (rng.uniform(networkList[layer],
				networkList[layer - 1]) + 0.01)*scaleFactor

		return fixedPointArray(weights, fixed_point_decimals)


	elif mode == "test":

		# Load weights from file
		with open(weightFile, 'rb') as fp:
			weights = np.load(fp)

		# Check if the weights are already in fixed point or not
		if trainPrecision == "fixedPoint":
			return weights

		elif trainPrecision == "float":
			return fixedPointArray(weights, fixed_point_decimals)

		else:
			# Invalid mode, print error and exit
			print('Invalid  training precision. Accepted values:\
			\n\t1) fixedPoint \n\t2) float')
			sys.exit()
	
	else:
		# Invalid mode, print error and exit
		print('Invalid operation mode. Accepted values:\n\t1) test\
		\n\t2) train')
		sys.exit()






def interLayerSynapses(network, synapseName, synapseWeight, layer):

	'''
	Initialize the inter layer synapses and add it to the network dictionary.

	INPUT:
		1) network: dictionary of the network.

		2) synapseName: string reporting the name of the connection. The
		standard names are "inh2exc" and "exc2inh". The function
		appends the number of the current layer.

		3) synapseWeight: float. Weight of the synapse. This is the same
		for all the connections within the layer.

		4) layer: index of the current layer. The count starts from 1.

	'''

	# Append the number of the current layer to the name
	synapseName = synapseName + str(layer)
	
	# Initialize the synapse weight
	network[synapseName] = {
		"weight" : synapseWeight
	}
