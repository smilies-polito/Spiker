mnistDir = "../../../../MNIST"

# Training
trainImages = mnistDir + "/train-images-idx3-ubyte"
trainLabels = mnistDir + "/train-labels-idx1-ubyte"

# Test
testImages = mnistDir + "/t10k-images-idx3-ubyte"
testLabels = mnistDir + "/t10k-labels-idx1-ubyte"

# Directory in which parameters and performance of the network are stored
paramDir = "./Parameters"

# Name of the parameters files
weightFilename = paramDir + "/weights"
thresholdFilename = paramDir + "/thresholds"
assignmentsFile = paramDir + "/assignments.npy"

# Name of the performance files
trainPerformanceFile = paramDir + "/trainPerformance.txt"
testPerformanceFile = paramDir + "/testPerformance.txt"

networkScript = "./runAccelerator.sh"
networkCompile = "./compileAccelerator.sh"

vhdlIoDir = "../../../../Vhdl/Hierarchical/Sim/IO"

imageFilename = vhdlIoDir + "/inputImage.txt"
countersFilename = vhdlIoDir + "/cntOut.txt"
