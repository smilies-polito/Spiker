# Directory in which the initialization files will be stored
hyperparametersDir = "../hyperparameters"

# Directory containing all the trained hyperparameters
trainedParamDir  = "../../../../python/custom/mark3_simplified/"\
			"parameters"

# Trained weights
weightsFilename = trainedParamDir + "/weights1.npy"

# Trained thresholds
inThreshFilename = trainedParamDir + "/thresholds1.npy"

# Root name of the BRAM initialization file. The index of the BRAM will be
# appended to it
bramRootFilename = hyperparametersDir + "/weights"

# Name of the threshold registers initialization file
outThreshFilename = hyperparametersDir + "/thresholds.init"
