vhdlDir = "../../../../Vhdl"
pythonDir = "../../../../Python"

# Directory in which the initialization files will be stored
parametersDir = vhdlDir + "/Hierarchical/Sim/Parameters"

# Directory containing all the trained hyperparameters
trainedParamDir  = pythonDir + "/Models/Custom/FullPrecision/Parameters"

# Trained weights
weightsFilename = trainedParamDir + "/weights1.npy"

# Trained thresholds
inThreshFilename = trainedParamDir + "/thresholds1.npy"

# Root name of the BRAM initialization file. The index of the BRAM will be
# appended to it
bramRootFilename = parametersDir + "/weights"

# Name of the threshold registers initialization file
outThreshFilename = parametersDir + "/thresholds.init"
