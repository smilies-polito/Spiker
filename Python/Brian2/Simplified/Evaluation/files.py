developmentDir = "../"

mnistDir = "../../../../MNIST"

# Training
trainImages = mnistDir + "/train-images-idx3-ubyte"
trainLabels = mnistDir + "/train-labels-idx1-ubyte"

# Test
testImages = mnistDir + "/t10k-images-idx3-ubyte"
testLabels = mnistDir + "/t10k-labels-idx1-ubyte"


# Directory containing equations and their parameters
equationsDir = "../Equations"


# Directory in which parameters and performance of the network are stored
paramDir = "./Parameters"


# Name of the parameters files
weightFilename = paramDir + "/weights"
thetaFilename = paramDir + "/theta"
assignmentsFile = paramDir + "/assignments.npy"


# Name of the performance files
trainPerformanceFile = paramDir + "/performance.txt"
testPerformanceFile = paramDir + "/testPerformance.txt"
