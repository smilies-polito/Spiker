import brian2 as b2
import numpy as np



def createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict):

	poissonGroup = b2.PoissonGroup(networkList[0], 0*b2.Hz)

	excLayersList, inhLayersList = createLayersStructure(networkList, 
	 				equationsDict, parametersDict)

	synapsesList = connectLayersStructure(networkList, poissonGroup, 
	 				excLayersList, inhLayersList,stdpDict, 
					weightInitDict)

	
	spikeMonitor = b2.SpikeMonitor(excLayersList[-1], record=False)

	return b2.Network(poissonGroup, excLayersList, inhLayersList, 
			synapsesList, spikeMonitor)





def createLayersStructure(networkList, equationsDict, parametersDict):

	networkSize = len(networkList)

	excLayersList = [0] * (networkSize- 1)
	inhLayersList = [0] * (networkSize - 1)

	for layer in range(1, networkSize):

		excLayersList[layer-1] = createLayer(
			networkList[layer],
			equationsDict["neuronsEqs_exc"],
			equationsDict["thresh_exc"],
			parametersDict["tRefrac_exc"],
			equationsDict["reset_exc"],
			"excLayer" + str(layer),
			parametersDict["vRest_exc"],
			"exc"
		)


		inhLayersList[layer-1] = createLayer(
			networkList[layer],
			equationsDict["neuronsEqs_inh"],
			equationsDict["thresh_inh"],
			parametersDict["tRefrac_inh"],
			equationsDict["reset_inh"],
			"inhLayer" + str(layer),
			parametersDict["vRest_inh"],
			"inh"
		)

	return excLayersList, inhLayersList






def createLayer(numberOfNeurons, neuronsEquations, threshEquations,
		refractoryPeriod, resetEquations, groupName, restPotential,
		neuronType):

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
		neuronGroup.theta = np.ones(numberOfNeurons)*20*b2.mV

	return neuronGroup







def connectLayersStructure(networkList, poissonGroup, excLayersList, 
			inhLayersList,stdpDict, weightInitDict):

	networkSize = len(networkList)
	synapsesList = [0] * (networkSize - 1) * 3

	for layer in range (1, networkSize):
		
		synapsesList[3*(layer - 1)] = exc2excConnection(
			networkList, 
			poissonGroup, 
			excLayersList, 
			stdpDict, 
			layer)


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
			layer):

	weightMatrix = (b2.random(networkList[layer]*networkList[layer - 1])
			+ 0.01)*0.3

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
