mnistDir = "../../../MNIST"

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


vhdlIoDir = "../../../Vhdl/mark3/sim/inputOutput"

inputFilename = vhdlIoDir + "/inputSpikes.txt"
outSpikesFilename = vhdlIoDir + "/pythonOutSpikes.txt"
membraneFilename = vhdlIoDir + "/pythonMembrane.txt"
countersFilename = vhdlIoDir + "/pythonCounters.txt"
