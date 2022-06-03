# Directpory containing the approximated python simulation of the model
pythonSrcDir = "../../../python/custom/mark4_finitePrecision"

# Functions to format the weights in order to easily initialize the brams
bramInitDir = "bramInit"

# Directory in which the initialization files will be stored
hyperparametersDir = "hyperparameters"

# Directory containing all the trained hyperparameters
trainedParamDir  = "../../../python/custom/mark4_finitePrecision/"\
			"parameters"

# Trained weights
weightsFilename = trainedParamDir + "/weights"

# Trained thresholds
inThreshFilename = trainedParamDir + "/thresholds"

# Trained assignments of the output layer
assignmentsFile = trainedParamDir + "/assignments.npy"

# Root name of the BRAM initialization file. The index of the BRAM will be
# appended to it
bramRootFilename = hyperparametersDir + "/weights"

# Name of the threshold registers initialization file
outThreshFilename = hyperparametersDir + "/thresholds.init"

# Complete path of the MNIST files
trainImages = "../../../mnist/train-images-idx3-ubyte"
trainLabels = "../../../mnist/train-labels-idx1-ubyte"
testImages = "../../../mnist/t10k-images-idx3-ubyte"
testLabels = "../../../mnist/t10k-labels-idx1-ubyte"

# Directory containing the input spikes and the monitored output quantities
inOutDir = "inputOutput"

# Input spikes
inputFilename = inOutDir + "/inputSpikes.txt"

# Python monitored output
pythonOutSpikes = inOutDir + "/pythonOutSpikes.txt"
pythonMembrane = inOutDir + "/pythonMembrane.txt"
