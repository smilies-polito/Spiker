import numpy as np

from functions import storeWeights, fixedPointArray, createDir
from files import *
from parameters import *

createDir(hyperparametersDir)

with open(weightsFilename, 'rb') as fp:
	weightsArray = np.load(fp)

i = 0
flag = 0

weightsArray = fixedPointArray(weightsArray, fixed_point_decimals)


while i < len(wordWidthsList) and flag == 0:

	if int(bramSize//wordWidthsList[i]) > numberOfInputs:

		wordWidth = wordWidthsList[i]
		flag = 1

	i += 1

subArraySize = int(wordWidth//bitWidth)

storeWeights(weightsArray, subArraySize, bitWidth, wordWidth, bramRootFilename)
