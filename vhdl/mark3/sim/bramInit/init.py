import numpy as np

from functions import storeWeights, fixedPointArray, createDir, formatAndStore
from files import *
from parameters import *


def bramInit(hyperparametersDir, weightsFilename, inThreshFilename,
		wordWidthsList, bramSize, numberOfInputs, weightsBitWidth,
		neuronsBitWidth, bramRootFilename, outThreshFilename):

	# Create the target directory in which the weights will be stored
	createDir(hyperparametersDir)

	# Import the weights in form of a numpy array
	with open(weightsFilename, 'rb') as fp:
		weightsArray = np.load(fp)

	# Import the thresholds in form of a numpy array
	with open(inThreshFilename, 'rb') as fp:
		threshArray = np.load(fp)


	# Quantize the weights to fixed point
	weightsArray = fixedPointArray(weightsArray, fixed_point_decimals)

	# Quantize the thresholds to fixed point
	threshArray = fixedPointArray(threshArray, fixed_point_decimals)


	i = 0
	flag = 0

	# Find the proper word-width for the BRAM between the available ones
	while i < len(wordWidthsList) and flag == 0:

		# Check if all the inputs can be stored with the selected
		# word-width
		if int(bramSize//wordWidthsList[i]) > numberOfInputs:

			# Set the corresponding word-width and exit
			wordWidth = wordWidthsList[i]
			flag = 1

		i += 1

	# Set the number of neurons that can be stored within the BRAM
	subArraySize = int(wordWidth//weightsBitWidth)

	# Store the weights in binary format to import them to initialize the
	# BRAMs
	storeWeights(weightsArray, subArraySize, weightsBitWidth, wordWidth,
			bramRootFilename, ".mem")
	with open(outThreshFilename, "w") as fp:

		for threshold in threshArray[0]:

			binaryValue = binaryFormat.format(threshold)

			fp.write(binaryValue)
			fp.write("\n")

bramInit(hyperparametersDir, weightsFilename, inThreshFilename,
		wordWidthsList, bramSize, numberOfInputs, weightsBitWidth,
		neuronsBitWidth, bramRootFilename, outThreshFilename)
