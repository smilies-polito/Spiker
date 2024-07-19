import subprocess
import timeit
import sys
import numpy as np

from snntorch import spikegen

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def createDir(dirName):

	"""
	Create a new directory. If it already exists it is firstly remove.

	INPUT:

		dirName: string. Name of the directory to create
	"""

	# Check if the directory exists
	cmdString = "if [[ -d " + dirName + " ]]; then "

	# If it exists remove it
	cmdString += "rm -r " + dirName + "; "
	cmdString += "fi; "

	# Create the directory
	cmdString += "mkdir " + dirName + "; "
	
	# Run the complete bash command
	sp.run(cmdString, shell=True, executable="/bin/bash")




def initAssignments(mode, networkList, assignmentsFile):

	"""
	Initialize the assignments of the output layer"s neurons.

	INPUT:
		
		1) mode: string. It can be "train" or "test".

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) assignmentsFile: string. Complete name of the file which
		contains the assignments of the output layer.

	"""

	if mode == "train":

		# Initialize assignments to value different from all the labels
		return -1*np.ones(networkList[-1])

	elif mode == "test":

		# Load the assignments from file	
		with open(assignmentsFile, "rb") as fp:
			return np.load(fp)

	else:
		print("Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train")
		sys.exit()


def seconds2hhmmss(seconds):

	"""
	Convert a time value from seconds to hh.mm.ss format

	INPUT:
		
		seconds: float number. Total amount of seconds to convert.

	OUTPUT:

		string containing the time expressed in hh.mm.ss format
	"""

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	seconds = int(seconds % 60)

	return str(hours) + "h " + str(minutes) + "min " + str(seconds) + "s"




def expDecay(dictionary, key, exp_shift, variable): 
	
	""" 
	Decrease the desired integer variable belonging to an entry of the dictionary
	with exponential decay.

	INPUT: 

		1) dictionary: generic dictionary of dictionaries.

		2) key: string. Name of the dictionary entry toupdate		

		3) exp_shift: bit shift for the exponential decay.

		4) variable: string. Name of the variable to update. This is the
		key of dictionary[key].
	"""

	dictionary[key][variable] -= dictionary[key][variable] >> exp_shift




def fixedPoint(value, fixed_point_decimals, bitwidth):

	"""
	Convert a value into fixed point notation.

	INPUT:

		1) value: floating point value to convert.

		2) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

	"""

	quant = int(value * 2**fixed_point_decimals)

	return saturate(quant, bitwidth)




def saturate(value, bitwidth):

	if value > 2**(bitwidth-1)-1:
		value = 2**(bitwidth-1)-1

	elif value < -2**(bitwidth-1):
		value = -2**(bitwidth-1)

	return value



def fixedPointArray(numpyArray, fixed_point_decimals, bitwidth):

	"""
	Convert a NumPy array into fixed point notation.

	INPUT:

		1) numpyArray: floating point array to convert.

		2) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

	"""

	numpyArray = numpyArray * 2**fixed_point_decimals
	numpyArray = numpyArray.astype(int)

	return saturateArray(numpyArray, bitwidth)

def saturateArray(numpyArray, bitwidth):

	"""
	Convert a NumPy array into fixed point notation.

	INPUT:

		1) numpyArray: floating point array to convert.

		2) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

	"""

	numpyArray[numpyArray > 2**(bitwidth-1)-1] = 2**(bitwidth-1)-1
	numpyArray[numpyArray < -2**(bitwidth-1)] = -2**(bitwidth-1)
	numpyArray = numpyArray.astype(int)

	return numpyArray


def checkBitWidth(numpyArray, bitWidth):

	"""
	Check that values inside NumPy array don"t exceed a threshold.

	INPUT:

		1) numpyArray: array of values to check.

		2) bitWidth: number of bits on which the neuron works.
	"""
	

	if (numpyArray > 2**(bitWidth-1)-1).any():
		print("Value too high")
		sys.exit()

	elif (numpyArray < -2**(bitWidth-1)).any():
		print("Value too low")
		sys.exit()

def run(network, networkList, spikesTrains, dt_tauDict, exp_shift, stdpDict,
		mode, constSums, neuron_bitWidth):

	"""
	Run the network over the duration of the input spikes
	trains.

	INPUT:
		
		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		3) spikesTrains: two-dimensional NumPy array. One training step
		for each row; one input for each column.

		4) dt_tauDict: dictionary containing the exponential constants
		of the excitatory and inhibitory membrane and of the 
		threshold .

		5) exp_shift: bit shift for the exponential decay.

		6) stdpDict: dictionary containing the STDP parameters.

		7) mode: string. It can be "train" or "test".

		8) neuron_bitWidth: number of bits on which the neuron works.

	OUTPUT:

		1) spikesCounter: NumPy array containing the total amount of 
		generate spike for each neuron.

		2) spikesMonitor: numpy array. Temporal evolution of the spikes.

		3) membraneMonitor: numpy array. Temporal evolution of the
		membrane potential
	"""

	lastLayerSize = networkList[-1]
	lastLayerIndex = len(networkList) - 1
	trainDuration = spikesTrains.shape[0]

	# Initialize the output spikes counter to 0
	spikesCounter = np.zeros((1, lastLayerSize))
	spikesMonitor_0 = np.zeros((trainDuration, networkList[1])).astype(bool)
	spikesMonitor_1 = np.zeros((trainDuration, lastLayerSize)).astype(bool)
	membraneMonitor_0 = np.zeros((trainDuration, networkList[1])).astype(int)
	membraneMonitor_1 = np.zeros((trainDuration, lastLayerSize)).astype(int)

	for i in range(trainDuration):

		# Train the network over a single step
		updateNetwork(networkList, network, spikesTrains[i], dt_tauDict,
				exp_shift, stdpDict, mode, neuron_bitWidth)

		spikesMonitor_0[i] = network["excLayer" +
				str(1)]["outSpikes"][0]

		spikesMonitor_1[i] = network["excLayer" +
				str(lastLayerIndex)]["outSpikes"][0]

		membraneMonitor_0[i] = network["excLayer" +
				str(1)]["v"][0]

		membraneMonitor_1[i] = network["excLayer" +
				str(lastLayerIndex)]["v"][0]


		# Update the output spikes counter
		spikesCounter[0][network["excLayer" +
			str(lastLayerIndex)]["outSpikes"][0]] += 1

	if mode == "train":
		# Normalize the weights
		normalizeWeights(network, networkList, constSums)
	
	return spikesCounter, spikesMonitor_0, spikesMonitor_1, \
		membraneMonitor_0, membraneMonitor_1





def updateNetwork(networkList, network, inputSpikes, dt_tauDict, exp_shift,
		stdpDict, mode, neuron_bitWidth):

	"""
	One training step update for the entire network.

	INPUT:

		1) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.

		2) network: dictionary of the network.

		3) inputSpikes: NumPy array. Value of the input spikes within a
		single training step, one for each neuron.
		
		4) dt_tauDict: dictionary containing the exponential constants
		of the excitatory and inhibitory membrane and of the 
		threshold .

		5) stdpDict: dictionary containing the STDP parameters.

		6) exp_shift: bit shift for the exponential decay.

		7) stdpDict: dictionary containing the STDP parameters.

		8) mode: string. It can be "train" or "test".

		9) neuron_bitWidth: number of bits on which the neuron works.

	"""	

	layerName = "excLayer1"

	# Update the first excitatory layer
	updateExcLayer(network, 1, exp_shift, inputSpikes, neuron_bitWidth[0])

	for layer in range(2, len(networkList)):

		layerName = "excLayer" + str(layer)
		
		# Update the excitatory layer
		updateExcLayer(network, layer, exp_shift,
			network["excLayer" + str(layer - 1)]["outSpikes"][0],
			neuron_bitWidth[layer-1])


def rest(network, networkList):

	"""
	Bring the network into a rest state.

	INPUT:

		1) network: dictionary of the network.

		2) networkList: list of integer numbers. Each element of the 
		list corresponds to a layer and identifies the number of nodes
		in that layer.
	"""

	for layer in range(1, len(networkList)):

		# Reset the membrane potential to the rest value
		network["excLayer" + str(layer)]["v"][0][:] = network["excLayer"
			+ str(layer)]["vRest"]



def updateExcLayer(network, layer, exp_shift, inputSpikes, neuron_bitWidth):

	"""
	One training step update of the excitatory layer.

	INPUT:
	
		1) network: dictionary of the network.

		2) layer: index of the current layer. The count starts from 1.

		3) exp_shift: bit shift for the exponential decay.
		
		4) inputSpikes: boolean NumPy array containing the input spikes.

		5) neuron_bitWidth: number of bits on which the neuron works.

	"""

	layerName = "excLayer" + str(layer)
	
	# Generate spikes if the potential exceeds the dynamic threshold	
	generateOutputSpikes(network, layerName)

	# Reset the potential of the active neurons	
	resetPotentials(network, layerName)

	# Exponentially decrease the membrane potential
	expDecay(network, layerName, exp_shift, "v")


	if np.sum(inputSpikes) != 0:

		# Update membrane potential with the spikes from the previous layer
		all2allUpdate(network, layerName, "exc2exc" + str(layer),
				inputSpikes, neuron_bitWidth)


	checkBitWidth(network["excLayer" + str(layer)]["v"],
			neuron_bitWidth)


def generateOutputSpikes(network, layerName): 

	""" 
	Generate the output spikes for all those neurons whose membrane
	potential exceeds the threshold (variable with the homeostasis).

	INPUT: 

		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.
	"""
	
	network[layerName]["outSpikes"][0] = network[layerName]["v"][0] > \
		network[layerName]["vThresh"]




def resetPotentials(network, layerName):

	""" 
	Reset the membrane potential of all those neurons whose membrane
	potential has exceeded the threshold. 

	INPUT:
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.

	"""


	network[layerName]["v"][0][network[layerName]["outSpikes"][0]] = \
	network[layerName]["v"][0][network[layerName]["outSpikes"][0]] - \
	network[layerName]["vThresh"][network[layerName]["outSpikes"][0]]



def all2allUpdate(network, layerName, synapseName, inputSpikes, bitwidth):

	""" 
	Update the membrane potential of a fully connected layer.

	INPUT:
		
		1) network: dictionary of the network.

		2) layerName: string. Complete name of the layer, including the
		index of the layer itself.

		3) synapseName: string. Complete name of the synapse, including the
		index of the synapse itself.

		4) inputSpikes: boolean NumPy array containing the input spikes.

	"""

	for i in range(len(inputSpikes)):

		network[layerName]["v"][0] = network[layerName]["v"][0] + \
		network[synapseName]["weights"][:, i] * inputSpikes[i]

		network[layerName]["v"][0] = saturateArray(network[layerName]["v"][0],
				bitwidth)


def createNetwork(networkList, weightFilename, thresholdFilename, mode,
			excDictList, scaleFactors, inh2excWeights,
			fixed_point_decimals, neuron_bitwidth, weights_bitwidth,
			trainPrecision, rng):

	"""
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

	OUTPUT:
		network: dictionary containing the initialized network.
	"""

	network = {}

	i = 0
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
				fixed_point_decimals, neuron_bitwidth[i],
				trainPrecision)

		# Create the excitatory to excitatory connection
		intraLayersSynapses(network, "exc2exc", mode, networkList,
				weightFile, layer, scaleFactors[layer-1],
				fixed_point_decimals, weights_bitwidth[i],
				trainPrecision, rng)

		i += 1

	return network






def createLayer(network, layerType, initDict, networkList, layer, mode,
		thresholdFile, fixed_point_decimals, neuron_bitwidth,
		trainPrecision):

	"""
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
	"""

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
						neuron_bitwidth,
						trainPrecision),

		# Initialize the output spikes
		"outSpikes"	: np.zeros((1,
			networkList[layer])).astype(bool),

		# Initialize the output spikes
		"inhSpikes"	: np.zeros((1,
			networkList[layer])).astype(bool)
	}






def initializeThreshold(mode, thresholdFile, initDict, numberOfNeurons,
		fixed_point_decimals, bitwidth, trainPrecision):

	"""
	Initialize the thresholds.

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

	The function initialize the thresholds depending on the mode in which
	the network will be run, train or test.

	"""

	if mode == "train":

		# Initialize the thresholds to a starting value
		return initDict["vThresh0"]*np.ones((1,
			numberOfNeurons)).astype(int)

	elif mode == "test":

		# Load thresholds values from file
		with open(thresholdFile, "rb") as fp: 
			thresholds = np.load(fp)

		# Check if the thresholds are already in fixed point or not
		if trainPrecision == "fixedPoint":
			return thresholds

		elif trainPrecision == "float":
			return fixedPointArray(thresholds, fixed_point_decimals,
					bitwidth)

		else:
			# Invalid mode, print error and exit
			print("Invalid  training precision. Accepted values:\
				\n\t1) fixedPoint \n\t2) float")
			sys.exit()

	else:

		# Invalid mode, print error and exit
		print("Invalid operation mode. Accepted values: \n\t1) test\
			\n\t2) train")
		sys.exit()





def intraLayersSynapses(network, synapseName, mode, networkList, weightFile,
			layer, scaleFactor, fixed_point_decimals,
			weights_bitwidth, trainPrecision, rng):

	"""	
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
	"""
	
	# Append the number of the current layer to the name
	synapseName = synapseName + str(layer)

	network[synapseName] = {

		# Initialize the synapses weights
		"weights"	: initializeWeights(mode, networkList,
					weightFile, layer, scaleFactor,
					fixed_point_decimals, weights_bitwidth,
					trainPrecision, rng),

		# Initialize the pre-synaptic trace
		"pre"		: np.zeros((1, networkList[layer - 1])),

		# Initialize the post-synaptic trace
		"post"		: np.zeros((networkList[layer], 1))	

	}







def initializeWeights(mode, networkList, weightFile, layer, scaleFactor,
		fixed_point_decimals, bitwidth, trainPrecision, rng):

	"""
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

	"""

	if mode == "train":

		# Randomly initialize the weights
		weights = (rng.uniform(networkList[layer],
				networkList[layer - 1]) + 0.01)*scaleFactor

		return fixedPointArray(weights, fixed_point_decimals, bitwidth)


	elif mode == "test":

		# Load weights from file
		with open(weightFile, "rb") as fp:
			weights = np.load(fp)

		# Check if the weights are already in fixed point or not
		if trainPrecision == "fixedPoint":
			return weights

		elif trainPrecision == "float":
			return fixedPointArray(weights, fixed_point_decimals,
					bitwidth)

		else:
			# Invalid mode, print error and exit
			print("Invalid  training precision. Accepted values:\
			\n\t1) fixedPoint \n\t2) float")
			sys.exit()
	
	else:
		# Invalid mode, print error and exit
		print("Invalid operation mode. Accepted values:\n\t1) test\
		\n\t2) train")
		sys.exit()

def imgToSpikeTrain(image, num_steps, rng):

	""" 
	Convert a black and white image into spike trains using the Poisson
	method.

	INPUT:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) dt: float. Time step duration, expressed in milliseconds. 

		3) trainingSteps: total amount of time steps associated to the
		pixels" spikes trains.

		4) inputIntensity: current value of the pixel"s intensity.

		5) rng: NumPy random generator.


	OUTPUT:

		Two-dimensional boolean NumPy array. Each row corresponds to a
		time step. Each column corresponds to a pixel.
	"""

	# Create two-dimensional array of random values
	random2D = rng.uniform(size = (num_steps, image.shape[0]))

	# Convert the image into spikes trains
	return poisson(image, random2D)


def poisson(image, random2D):

	""" 
	Poisson convertion of the numerical values of the pixels into spike
	trains. 

	INPUT PARAMETERS:

		1) image: NumPy array containing the value of each pixel
		expressed as an integer.

		2) dt: time step duration, expressed in milliseconds. 

		3) random2D: teo-dimensional NumPy array containing random
		values between pixelMin and pixelMax.

		4) inputIntensity: current value of the pixel"s intensity.


	OUTPUT:

		Boolean two-dimensional array containing one spikes"
		train for each pixel.  
	"""

	# Create the boolean array of spikes with Poisson distribution
	return image[:] > random2D


weights_bitWidth	= [4, 4]
neuron_bitWidth		= [6, 6]
fixed_point_decimals 	= 4
exp_shift		= 10

# Directory in which parameters and performance of the network are stored
paramDir = "./Parameters128"

# Name of the parameters files
weightFilename = paramDir + "/weights"
thresholdFilename = paramDir + "/thresholds"
assignmentsFile = paramDir + "/assignments.npy"

in_spikes_file	= "in_spikes.txt"
out_spikes_file	= "out_spikes_python.txt"

data_dir ='./data/mnist'

image_height		= 28
image_width		= 28
image_index		= 13

# dataloader arguments
batch_size = 1

dtype = torch.float

# Network Architecture
num_inputs = image_width*image_height
num_hidden = 128
num_outputs = 10

# Temporal Dynamics
num_steps = 100
beta = 0.9375
tauExc = 100


num_epochs = 30
loss_hist = []
test_loss_hist = []
counter = 0

# List of layer sizes
networkList = [784, 128, 10]

mode = "test"
trainPrecision = "float"

# Excitatory layer
excDict = {			# Shifted values to minimize the interval
				# ----------------------------------------------
				# |Original|	|  Ref  |	|Shift |
				# |--------|----|-------|-------|------|
	"vRest"		: 0,	# | -65.0  | =	| -65.0 | + 	| 0.0  | mV
	"vReset"	: 0,	# | -60.0  | =	| -65.0 | + 	| 5.0  | mV
	"vThresh0"	: 1,	# | -52.0  | =	| -65.0 | + 	| 13.0 | mV
	"vThreshPlus"	: 0.05,	# |  0.05  | =	|   -   | + 	|  -   | mV
}


# Finite precision
excDict["vRest"] = fixedPoint(excDict["vRest"], fixed_point_decimals,
		neuron_bitWidth[0])
excDict["vReset"] = fixedPoint(excDict["vReset"], fixed_point_decimals,
		neuron_bitWidth[0])
excDict["vThresh0"] = fixedPoint(excDict["vThresh0"], fixed_point_decimals,
		neuron_bitWidth[0])
excDict["vThreshPlus"] = fixedPoint(excDict["vThreshPlus"],
				fixed_point_decimals, neuron_bitWidth[0])

# List of dictionaries of parameters for the layers
excDictList = [excDict] * (len(networkList) - 1)

# Training and resting periods in milliseconds
trainDuration = 350	# ms

# Array of scale factors for the random generation of the weights
scaleFactors = np.ones(len(networkList) - 1)

# Time step duration in milliseconds
dt = 2**-4 * tauExc		# ms


# Exponential time constants
dt_tauDict = {
	"exc" 		: dt/tauExc,
}

# Update and print intervals expressed in number of images
updateInterval = 10
printInterval = 250

# Initial intensity of the input pixels
startInputIntensity = 2.
inputIntensity = startInputIntensity


# Initialize history of accuracies
accuracies = []

# Initialize history of spikes
spikesEvolution = np.zeros((updateInterval, networkList[-1]))


# Minimum acceptable number of output spikes generated during the training.
countThreshold = 5

# NumPy default random generator.
rng = np.random.default_rng()



# Define a transform
transform = transforms.Compose([
	transforms.Resize((image_width, image_height)),
	transforms.Grayscale(),
	transforms.ToTensor(),
	transforms.Normalize((0,), (1,))]
)

test_set = datasets.MNIST(root=data_dir, train=False, download=True,
		transform=transform)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
		drop_last=True)


test_data = list(iter(test_loader))

# Create the network data structure
net = createNetwork(networkList, weightFilename, thresholdFilename, mode, 
			excDictList, scaleFactors, None,
			fixed_point_decimals, neuron_bitWidth, weights_bitWidth,
			trainPrecision, rng)

np.set_printoptions(threshold = np.inf)

with open("weights.txt", "w") as fp:
	for i in range(net["exc2exc1"]["weights"].shape[0]):
		fp.write(str(net["exc2exc1"]["weights"][i]).replace("\n", " "))
		fp.write("\n")


image = test_data[image_index][0].view(batch_size, -1).numpy()
label = int(test_data[image_index][1].int())

spikesTrains = imgToSpikeTrain(image, num_steps, rng)

with open(in_spikes_file, "w") as fp:
	for inputs in spikesTrains:
		fp.write((str(inputs.astype(int))[-2:0:-1].replace(" ",
			"").replace("\n", "")))
		fp.write("\n")

_, spikes_1, out_spikes, m0, m1 = run(net, networkList, spikesTrains,
		dt_tauDict, exp_shift, None, mode, None,
		neuron_bitWidth)


# print(m1[:, 1])
# print(out_spikes[:, 1])

with open(out_spikes_file, "w") as fp:
	for output in out_spikes:
		fp.write((str(output.astype(int))[1:-1].replace(" ", "")))
		fp.write("\n")
