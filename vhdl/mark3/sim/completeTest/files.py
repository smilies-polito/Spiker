# Complete path of the MNIST files
trainImages = "../../../../mnist/train-images-idx3-ubyte"
trainLabels = "../../../../mnist/train-labels-idx1-ubyte"
testImages = "../../../../mnist/t10k-images-idx3-ubyte"
testLabels = "../../../../mnist/t10k-labels-idx1-ubyte"

# Directory in which parameters and performance of the network are stored
paramDir = "./parameters"

# Name of the parameters files
weightFilename = paramDir + "/weights"
thresholdFilename = paramDir + "/thresholds"
assignmentsFile = paramDir + "/assignments.npy"

# Name of the performance files
trainPerformanceFile = paramDir + "/trainPerformance.txt"
testPerformanceFile = paramDir + "/testPerformance.txt"

networkScript = "./runAccelerator.sh"
networkCompile = "./compileAccelerator.sh"

vhdlIoDir = "../inputOutput"

spikesFilename = vhdlIoDir + "/inputSpikes.txt"
countersFilename = vhdlIoDir + "/cntOut.txt"
