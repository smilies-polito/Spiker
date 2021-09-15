import numpy as np
import sys


def createNetwork(networkList, weightFilename, thetaFilename, mode,
			excDictList, inhDictList, scaleFactors, exc2inhWeights,
			inh2excWeights):

	'''
	Create the complete network dictionary.

	INPUT:

		1) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) weightFilename: string. Root of the weights file name of
		each layer. The function appends the number of the current 
		layer to it.

		3) thetaFilename: string. Root of the theta file name of
		each layer. The function appends the number of the current 
		layer to it.

		4) mode: string. It can be "train" or "test".

		5) excDictList: list of dictionaries, each containing the 
		initialization values for a specific excitatory layer.

		6) inhDictList : list of dictionaries, each containing the 
		initialization values for a specific nhibitory layer.

		7) scaleFactors: float NumPy array. Factor used to scale the
		randomly generated weights for each layer. Needed in training
		mode. In test mode "None" can be used.

		8) exc2inhWeights: float NumPy array. Weight of the synapse for
		each layer. This is the same for all the connections within the layer.

		9) inh2excWeights: float NumPy array. Weight of the synapse for
		each layer. This is the same for all the connections within the layer.

	'''

	network = {}

	for layer in range(1, len(networkList)):

		if mode == "test":
			weightFile = weightFilename + str(layer)
			thetaFile = thetaFilename + str(layer)
		else:
			weightFile = None
			thetaFile = None

		# Create the excitatory layer
		createLayer(network, "exc", excDictList[layer-1], networkList,
				layer, mode, thetaFile)

		# Create the inhibitory layer
		createLayer(network, "inh", inhDictList[layer-1], networkList, 
				layer, mode, thetaFile)

		# Create the excitatory to excitatory connection
		intraLayersSynapses(network, "exc2exc", mode, networkList,
				weightFile, layer, scaleFactors[layer-1])

		# Create the excitatory to inhibitory connection
		interLayerSynapses(network, "exc2inh", exc2inhWeights[layer-1],
					layer)

		# Create the inhibitory to excitatory connection
		interLayerSynapses(network, "inh2exc", inh2excWeights[layer-1],
					layer)

	return network






def createLayer(network, layerType, initDict, networkList, layer, mode,
		thetaFile):

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

		7) thetaFile: complete name of the file containing the theta 
		parameters for the current layer.
	
	'''

	# Create the name for the layer
	layerName = layerType + "Layer" + str(layer)

	network[layerName] = {

		# Initialize the membrane potentials at the reset voltage		
		"v"		: initDict["vReset"] * np.ones((1, 
					networkList[layer])),

		# Initialize the threshold potential
		"vThresh"	: initDict["vThresh"],

		# Initialize the rest potential
		"vRest"		: initDict["vRest"],

		# Initialize the reset potential
		"vReset"	: initDict["vReset"],

		# Initialize the output spikes
		"outSpikes"	: np.zeros((1, networkList[layer])).astype(bool)
	}

	if layerType == "exc":

		# Initialize the homeostasis parameter
		network[layerName]["thetaPlus"] = initDict["thetaPlus"]

		# Initialize the dynamic homeostasis
		network[layerName]["theta"] = initializeTheta(mode, thetaFile,
						initDict, networkList[layer])





def initializeTheta(mode, thetaFile, initDict, numberOfNeurons):

	'''
	Initialize the dynamic homeostasis parameter theta.

	INPUT:

		1) mode: string. It can be "train" or "test".

		2) thetaFile: complete name of the file containing the theta 
		parameters for the current layer.

		3) initDict: dictionary containing the initialization values for
		the the layer.

		4) numberOfNeurons: number of neurons in the layer.


	The function initialize the theta parameter depending on the mode in
	which the network will be run, train or test.

	'''

	if mode == "train":

		# Initialize theta to a starting value
		return np.ones((1, numberOfNeurons))*initDict["initTheta"]

	elif mode == "test":

		# Load theta values from file
		with open(thetaFile, 'rb') as fp: 
			return np.load(fp)
	else:

		# Invalid mode, print error and exit
		print('Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train')
		sys.exit()





def intraLayersSynapses(network, synapseName, mode, networkList, weightFile,
			layer, scaleFactor):

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
		
	'''
	
	# Append the number of the current layer to the name
	synapseName = synapseName + str(layer)

	network[synapseName] = {

		# Initialize the synapses weights
		"weights"	: initializeWeights(mode, networkList, 
					weightFile, layer, scaleFactor),

		# Initialize the pre-synaptic trace
		"pre"		: np.zeros((1, networkList[layer - 1])),

		# Initialize the post-synaptic trace
		"post"		: np.zeros((networkList[layer], 1))	

	}







def initializeWeights(mode, networkList, weightFile, layer, scaleFactor):

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


	The function initializes the weights depending on the mode in
	which the network will be run, train or test.

	'''

	if mode == "train":

		# Randomly initialize the weights
		rng = np.random.default_rng()
		return (rng.random((networkList[layer], networkList[layer - 1]))
			+ 0.01)*scaleFactor

	elif mode == "test":

		# Load weights from file
		with open(weightFile, 'rb') as fp:
			return np.load(fp)
	
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
