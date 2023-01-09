import brian2 as b2
import numpy as np
import sys

# Functions to create and initialize the Spiking Neural Network

def createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename,
		scaleFactors):

	"""
	Create a Brian 2 simulation of a Spiking Neural Network:

	INPUT:

		1) networkList: list. Contains the network structure. Each
		element contain the number of neurons of the corresponding
		layer.

		2) equationsDict: dictionary. Contains the model equations.

		3) parametersDict: dictionary. Contains the neurons' internal
		parameters.

		4) stdpDict: dictionary. Contains the STDP parameters. Not
		required if mode = "test"

		5) weightInitDict: dictionary. Contains the initialization
		values for the weights. Not required if mode = "test"

		6) mode: string. Can be "train" or "test"

		7) thetaFilename: string. Name of the file containing the
		pre-trained thresholds. Not required if mode = "train"

		8) weightFilename: string. Name of the file containing the
		pre-trained weights. Not required if mode = "train"

		9) scaleFactors: list. Each element corresponds to a layer.
		Allows to scale the values of the weights during the random
		initialization. Not required if mode = "test"
		

	OUTPUT:

		Brian 2 network object containing the initialized network. 
	"""

	# Create the input layer to convert inputs into spikes
	poissonGroup = b2.PoissonGroup(networkList[0], 0*b2.Hz)

	# Create the excitatory and inhibitory layers
	excLayersList, inhLayersList = createLayersStructure(networkList, 
	 				equationsDict, parametersDict, mode,
					thetaFilename)

	# Interconnect the layers
	synapsesList = connectLayersStructure(networkList, poissonGroup, 
	 				excLayersList, inhLayersList,stdpDict, 
					weightInitDict, mode, weightFilename,
					scaleFactors)

	# Monitor the spikes evolution
	spikeMonitor = b2.SpikeMonitor(excLayersList[-1], record=False)

	# Return the initialized network
	return b2.Network(poissonGroup, excLayersList, inhLayersList, 
			synapsesList, spikeMonitor)






def createLayersStructure(networkList, equationsDict, parametersDict, mode,
			thetaFilename):

	"""
	Create excitatory and inhibitory layers.

	INPUT:

		1) networkList: list. Contains the network structure. Each
		element contain the number of neurons of the corresponding
		layer.

		2) equationsDict: dictionary. Contains the model equations.

		3) parametersDict: dictionary. Contains the neurons' internal
		parameters.

		4) mode: string. Can be "train" or "test"

		5) thetaFilename: string. Name of the file containing the
		pre-trained thresholds. Not required if mode = "train"

	OUTPUT:

		1) excLayersList: list. Contains the excitatory layers in form
		of Brian 2 NeuronGroup objects.

		2) inhLayersList: list. Contains the inhibitory layers in form
		of Brian 2 NeuronGroup objects.
	"""

	# Total number of layers
	networkSize = len(networkList)

	excLayersList = [0 for i in range(networkSize - 1)]
	inhLayersList = [0 for i in range(networkSize - 1)]

	for layer in range(1, networkSize):

		thetaFile = thetaFilename + str(layer) + ".npy"

		excLayersList[layer-1] = createLayer(
			networkList[layer],
			equationsDict["neuronsEqs_exc"],
			equationsDict["thresh_exc"],
			parametersDict["tRefrac_exc"],
			equationsDict["reset_exc"],
			"excLayer" + str(layer),
			parametersDict["vRest_exc"],
			"exc",
			mode,
			thetaFile
		)


		inhLayersList[layer-1] = createLayer(
			networkList[layer],
			equationsDict["neuronsEqs_inh"],
			equationsDict["thresh_inh"],
			parametersDict["tRefrac_inh"],
			equationsDict["reset_inh"],
			"inhLayer" + str(layer),
			parametersDict["vRest_inh"],
			"inh",
			mode,
			thetaFile
		)

	return excLayersList, inhLayersList






def createLayer(numberOfNeurons, neuronsEquations, threshEquations,
		refractoryPeriod, resetEquations, groupName, restPotential,
		neuronType, mode, thetaFile):

	# Create the group
	neuronGroup = b2.NeuronGroup(
		numberOfNeurons, 
		neuronsEquations, 
		threshold = threshEquations, 
		refractory = refractoryPeriod, 
		reset = resetEquations, 
		method = 'euler',
		name = groupName
	)

	# Initialize the membrane potential
	neuronGroup.v = restPotential - 40.*b2.mV

	# Initialize the threshold parameter theta
	if neuronType == "exc":
		initializeTheta(neuronGroup, numberOfNeurons, mode, thetaFile)

	return neuronGroup




def initializeTheta(neuronGroup, numberOfNeurons, mode, thetaFile):

	if mode == "train":
		neuronGroup.theta = np.ones(numberOfNeurons)*20*b2.mV
	elif mode == "test":
		with open(thetaFile, 'rb') as fp: 
			neuronGroup.theta = np.load(fp)*b2.mV
	else:
		print('Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train')
		sys.exit()




def connectLayersStructure(networkList, poissonGroup, excLayersList, 
			inhLayersList,stdpDict, weightInitDict, mode,
			weightFilename, scaleFactors):

	networkSize = len(networkList)
	synapsesList = [0] * (networkSize - 1) * 3

	for layer in range (1, networkSize):
		
		weightFile = weightFilename + str(layer) + ".npy"
		
		synapsesList[3*(layer - 1)] = exc2excConnection(
			networkList, 
			poissonGroup, 
			excLayersList, 
			stdpDict, 
			layer,
			mode,
			weightFile,
			scaleFactors[layer-1])


		synapsesList[3*(layer -1 ) + 1] = connectLayers(
			excLayersList[layer - 1], 
			inhLayersList[layer - 1], 
			synapseModel = "w : 1", 
			onPreEquation = "ge = ge + w",
			onPostEquation = None, 
			connectionType = "i==j", 
			weightInit = weightInitDict["exc2inh"], 
			name = "exc2inh" + str(layer))


		synapsesList[3*(layer - 1) + 2] = connectLayers(
			inhLayersList[layer - 1], 
			excLayersList[layer - 1],
			synapseModel = "w : 1", 
			onPreEquation = "gi = gi + w",
			onPostEquation = None, 
			connectionType = "i!=j", 
			weightInit = weightInitDict["inh2exc"], 
			name = "inh2exc" + str(layer))


	return synapsesList







def exc2excConnection(networkList, poissonGroup, excLayersList, stdpDict, 
			layer, mode, weightFile, scaleFactor):

	weightMatrix = initializeWeights(mode, networkList, weightFile, layer,
			scaleFactor)

	if layer == 1:

		exc2exc = connectLayers(
			poissonGroup, 
			excLayersList[layer-1], 
			stdpDict["stdpEqs"], 
			stdpDict["stdpPre"],
			stdpDict["stdpPost"], 
			connectionType = "i==j or i!=j", 
			weightInit = weightMatrix, 
			name = "poisson2exc"
		)

	else:

		exc2exc = connectLayers(
			excLayersList[layer - 2], 
			excLayersList[layer - 1], 
			stdpDict["stdpEqs"], 
			stdpDict["stdpPre"],
			stdpDict["stdpPost"], 
			connectionType = "i==j or i!=j", 
			weightInit = weightMatrix, 
			name = "exc2exc" + str(layer)
		)

	return exc2exc






def initializeWeights(mode, networkList, weightFile, layer, scaleFactor):

	if mode == "train":
		return (b2.random(networkList[layer]*networkList[layer - 1])
			+ 0.01)*scaleFactor

	elif mode == "test":
		with open(weightFile, 'rb') as fp:
			return np.load(weightFile)
	
	else:
		print('Invalid operation mode. Accepted values:\n\t1) test\
		\n\t2) train')
		sys.exit()



def connectLayers(originGroup, targetGroup, synapseModel, onPreEquation,
		onPostEquation, connectionType, weightInit, name):
	
	# Create the synapse
	synapse = b2.Synapses(
		originGroup, 
		targetGroup,  
		model = synapseModel,
		on_pre = onPreEquation,
		on_post = onPostEquation,
		name = name
	)

	# Create the connection
	synapse.connect(connectionType)

	# Initialize the weights
	synapse.w = weightInit

	return synapse
