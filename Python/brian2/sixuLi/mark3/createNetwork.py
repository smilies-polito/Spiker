import brian2 as b2
import numpy as np
import sys


from equations import excInhConnectionDict

from equationsParameters import *



def createNetwork(networkList, equationsDict, parametersDict, stdpDict,
		weightInitDict, mode, thetaFilename, weightFilename,
		scaleFactors):

	'''
	Create a complete brian2 network object with the desired
	values for the hyperparameters.

	INPUT:
		1) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) equationsDict: dictionary containing the membrane potential
		equations. See equations.py for more details.

		3) parametersDict: dictionary containing the parameters of the
		neurons. See equationPrameters.py for more details.

		4) stdpDict: dictionary containing the stdp equations. See
		trainEquations.py for more details.

		5) weightInitDict: dictionary containing the constant values to
		which the connections from excitatory to inhibitory neurons and
		viceversa will be initialize.

		6) mode: string. It can be "train" or "test".

		7) thetaFilename: name of the file containing the theta
		parameters after a complete training. Needed for test mode.

		8) weightFilename: name of the file containing the values of the
		weights after a complete training. Needed for test mode.

		9) scaleFactors: NumPy array with a number of elements equal to
		the number of layers. Each element represent the scale factor
		used in the generation of the random weights for each layer.

	

	RETURN VALUE:

		Brian2 network object with the required number of excitatory and
		inhibitory layers properly interconnected. 
	'''


	# Create the Poisson input layer
	poissonGroup = b2.PoissonGroup(networkList[0], 0*b2.Hz)

	# Create the excitatory and inhibitory layers of the network
	excLayersList, inhLayersList = createLayersStructure(networkList, 
	 				equationsDict, parametersDict, mode,
					thetaFilename)

	# Interconnect the various layers
	synapsesList = connectLayersStructure(networkList, poissonGroup, 
	 				excLayersList, inhLayersList,stdpDict, 
					weightInitDict, mode, weightFilename,
					scaleFactors)

	# Create the monitor for the output spikes
	spikeMonitor = b2.SpikeMonitor(excLayersList[-1], record = False)

	# Create the complete network putting together all the components
	return b2.Network(poissonGroup, excLayersList, inhLayersList, 
			synapsesList, spikeMonitor)







def createLayersStructure(networkList, equationsDict, parametersDict, mode,
			thetaFilename):


	'''
	Create all the excitatory and inhibitory layers of the nertwork.

	INPUT:	
		1) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) equationsDict: dictionary containing the membrane potential
		equations. See equations.py for more details.

		3) parametersDict: dictionary containing the parameters of the
		neurons. See equationPrameters.py for more details.

		4) mode: string. It can be "train" or "test".

		5) thetaFilename: name of the file containing the theta
		parameters after a complete training. Needed for test mode.

	RETURN VALUES:

		1) excLayersList: list of Brian2 NeuronGroup objects, one for
		each excitatory layer.

		2) inhLayersList: list of Brian2 NeuronGroup objects, one for
		each inhibitory layer.
	'''

	networkSize = len(networkList)

	# Preallocate the two lists
	excLayersList = [0] * (networkSize- 1)
	inhLayersList = [0] * (networkSize - 1)

	for layer in range(1, networkSize):

		# Complete name of the file containing the theta parameters for
		# the current layer
		thetaFile = thetaFilename + str(layer) + ".npy"

		# Create the excitatory layer
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

		# Create the inhibitory layer
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
		refractoryPeriod, resetEquations, groupName, resetPotential,
		neuronType, mode, thetaFile):

	'''
	Create a single layer of neurons.

	INPUT:
		1) numberOfNeurons: number of neurons in the layer.

		2) neuronsEquations: multiline string or Brian2 equation object 
		describing the membrane potential's temporal evolution.
		
		3) threshEquations: multiline string or Brian2 equation object
		describing the condition for the threshold to be exceeded.

		4) refractoryPeriod: float number. Period after the generation 
		of a spike after which the neurons stay inactive.

		5) resetEquations: multiline string or Brian2 equation object 
		describing the operations performed to reset the neuron's state.

		6) groupName: generic string used to identify the group. The
		number of the layer will be appended to form the complete
		layer's name. The standard names used by the training/test
		functions are "excLayer" and "inhLayer".

		7) resetPotential: float value. Reset value for the membrane 
		potential.

		8) neuronType: string. It can be "exc" or "inh".

		9) mode: string. It can be "train" or "test".

		10) thetaFile: complete name of the file containing the theta 
		parameters for the current layer


	RETURN VALUE:

		neuronGroup: Brian2 object corresponding to a properly
		initialized layer of neurons.
	'''

	# Create the layer
	neuronGroup = b2.NeuronGroup(
		numberOfNeurons, 
		neuronsEquations, 
		threshold = threshEquations, 
		refractory = refractoryPeriod, 
		reset = resetEquations, 
		method = 'exact',
		name = groupName
	)

	# Initialize the membrane potentials
	neuronGroup.v = resetPotential

	# Initialize the threshold parameter theta
	if neuronType == "exc":
		initializeTheta(neuronGroup, numberOfNeurons, mode, thetaFile)

	return neuronGroup




def initializeTheta(neuronGroup, numberOfNeurons, mode, thetaFile):

	'''
	Initialize the variable threshold parameter theta.

	INPUT:

		1) neuronGroup: Brian2 object corresponding to a properly
		initialized layer of neurons.

		2) numberOfNeurons: number of neurons in the layer.

		3) mode: string. It can be "train" or "test".

		4) thetaFile: complete name of the file containing the theta 
		parameters for the current layer.

	The function initialize the theta parameter depending on the mode in
	which the network will be run, train or test.

	'''

	if mode == "train":

		# Initialize theta to a starting value
		neuronGroup.theta = np.ones(numberOfNeurons)*20

	elif mode == "test":

		# Load theta values from file
		with open(thetaFile, 'rb') as fp: 
			neuronGroup.theta = np.load(fp)
	else:

		# Invalid mode, print error and exit
		print('Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train')
		sys.exit()




def connectLayersStructure(networkList, poissonGroup, excLayersList, 
			inhLayersList, stdpDict, weightInitDict, mode,
			weightFilename, scaleFactors):

	'''
	Connect the previously created layers to form a complete network.

	INPUT:
		1) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) poissonGroup: Brian2 specific Poisson object.

		3) excLayersList: list of Brian2 NeuronGroup objects, one for
		each excitatory layer.

		4) inhLayersList: list of Brian2 NeuronGroup objects, one for
		each inhibitory layer.

		5) stdpDict: dictionary containing the stdp equations. See
		trainEquations.py for more details.

		6) weightInitDict: dictionary containing the constant values to
		which the connections from excitatory to inhibitory neurons and
		viceversa will be initialize.

		7) mode: string. It can be "train" or "test".

		8) weightFilename: name of the file containing the values of the
		weights after a complete training. Needed for test mode.

		9) scaleFactors: NumPy array with a number of elements equal to
		the number of layers. Each element represent the scale factor
		used in the generation of the random weights for each layer.


	RETURN VALUE:

		synapseList: list of Brian2 Synapse objects, one for each
		connection in the network.

	
	Note that the function uses standard names for the connections, in
	particular "inh2exc" and "exc2inh" + the number of the current
	layer.

	'''

	networkSize = len(networkList)

	# Preallocate the list
	synapsesList = [0] * (networkSize - 1) * 3

	for layer in range (1, networkSize):
		
		# Complete name of the file containing the trained weights for
		# the current layer
		weightFile = weightFilename + str(layer) + ".npy"
		
		# Connect the excitatory layer to the previous one
		synapsesList[3*(layer - 1)] = exc2excConnection(
			networkList, 
			poissonGroup, 
			excLayersList, 
			stdpDict, 
			layer,
			mode,
			weightFile,
			scaleFactors[layer-1])

		# Connect excitatory to inhibitory
		synapsesList[3*(layer -1 ) + 1] = connectLayers(
			excLayersList[layer - 1], 
			inhLayersList[layer - 1], 
			synapseModel = excInhConnectionDict["exc2inhEqs"],
			onPreEquation = excInhConnectionDict["exc2inhPre"],
			onPostEquation = None, 
			connectionType = "i==j", 
			weightInit = weightInitDict["exc2inh"], 
			name = "exc2inh" + str(layer))

		# Connect inhibitory to excitatory
		synapsesList[3*(layer - 1) + 2] = connectLayers(
			inhLayersList[layer - 1], 
			excLayersList[layer - 1],
			synapseModel = excInhConnectionDict["inh2excEqs"], 
			onPreEquation = excInhConnectionDict["inh2excPre"],
			onPostEquation = None, 
			connectionType = "i!=j", 
			weightInit = weightInitDict["inh2exc"], 
			name = "inh2exc" + str(layer))


	return synapsesList







def exc2excConnection(networkList, poissonGroup, excLayersList, stdpDict, 
			layer, mode, weightFile, scaleFactor):

	'''
	Connect two excitatory layers.

	INPUT:
		
		1) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) poissonGroup: Brian2 specific Poisson object.

		3) excLayersList: list of Brian2 NeuronGroup objects, one for
		each excitatory layer.

		4) stdpDict: dictionary containing the stdp equations. See
		trainEquations.py for more details.

		5) layer: number of the current layer. The count starts from 1.

		6) mode: string. It can be "train" or "test".

		7) weightFile: Complete name of the file containing the trained 
		weights for the current layer

		8) scaleFactor: float number. Factor used to scale the randomly
		generated weights.


	RETURN VALUE:

		exc2exc: Brian2 Synapse object connecting two excitatory layers.

	Note that the function uses standard names for the connections, in
	particular "poisson2exc" for the connection between the input layer and
	the first excitatory layer and "exc2exc" + the number of the current
	layer for all the others.

	'''

	# Initialize weights
	weightMatrix = initializeWeights(mode, networkList, weightFile, layer,
			scaleFactor)

	if layer == 1:

		# Connect with the input Poisson layer
		exc2exc = connectLayers(
			poissonGroup, 
			excLayersList[layer-1], 
			stdpDict["stdpEqs"], 
			stdpDict["stdpPre"],
			stdpDict["stdpPost"], 
			connectionType = "i==j or i!=j", 
			weightInit = weightMatrix, 
			name = "exc2exc" + str(layer)
		)

	else:

		# Connect with the previous excitatory layer
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

	# Initialize the stdp parameters pre and post
	if mode == "train":
		exc2exc.pre = 0
		exc2exc.post = 0

	return exc2exc






def initializeWeights(mode, networkList, weightFile, layer, scaleFactor):

	'''
	Initialize the weights of the connections between two layers.

	INPUT:	
		1) mode: string. It can be "train" or "test".
		
		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) weightFile: Complete name of the file containing the trained
		weights for the current layer

		4) layer: index of the current layer. The count starts from 1.

		5) scaleFactor: float number. Factor used to scale the randomly
		generated weights.

	The function initialize the weights depending on the mode in
	which the network will be run, train or test.

	'''


	if mode == "train":

		# Randomly initialize the weights
		return (b2.random(networkList[layer]*networkList[layer - 1])
			+ 0.01)*scaleFactor

	elif mode == "test":

		# Load weights from file
		with open(weightFile, 'rb') as fp:
			return np.load(weightFile).reshape(784*400)
	
	else:
		# Invalid mode, print error and exit
		print('Invalid operation mode. Accepted values:\n\t1) test\
		\n\t2) train')
		sys.exit()




def connectLayers(originGroup, targetGroup, synapseModel, onPreEquation,
		onPostEquation, connectionType, weightInit, name):

	'''
	Connect two generic layers, that are two Brian2 NeuronGroup objects.

	INPUT:
		1) originGroup: Brian2 NeuronGroup object. Starting group for
		the connection.

		2) targetGroup: Brian2 NeuronGroup object. Ending group for the
		connection.

		3) synapseModel: multiline string or Brian2 equation object 
		describing the behaviour of the synapse.

		4) onPreEquation: multiline string or Brian2 equation object 
		describing what happens when an input spike arrives.

		5) onPostEquation: multiline string or Brian2 equation object 
		describing what happens when an output spike is generated.

		6) connectionType: string that explains how thw neurons will be
		connected. It can be for example "i==j" if the neurons need to
		be connected in a one to one fashion.

		7) weightInit: initialization values for the connection weights.

		8) name: string identifying the name of the synapse. The names
		used by the previous functions, for example for the first layer
		are: "poisson2exc", "exc2exc1", "exc2inh1" and "inh2exc1"
 
	'''
	
	# Create the synapse between origin group and target group
	synapse = b2.Synapses(
		originGroup, 
		targetGroup,  
		model = synapseModel,
		on_pre = onPreEquation,
		on_post = onPostEquation,
		name = name
	)

	# Connect the two elements with the desired association between neurons
	synapse.connect(connectionType)

	# Set the weights to the initialized values
	synapse.w = weightInit

	return synapse
