# Complete path of the MNIST files
trainImages = "../../brian2/mnist/train-images-idx3-ubyte"
trainLabels = "../../brian2/mnist/train-labels-idx1-ubyte"
testImages = "../../brian2/mnist/t10k-images-idx3-ubyte"
testLabels = "../../brian2/mnist/t10k-labels-idx1-ubyte"

# Directory in which parameters and performance of the network are stored
paramDir = "./parameters"

# Name of the parameters files
weightFilename = paramDir + "/weights"
thresholdFilename = paramDir + "/thresholds"
assignmentsFile = paramDir + "/assignments.npy"

# Name of the performance files
trainPerformanceFile = paramDir + "/trainPerformance.txt"
testPerformanceFile = paramDir + "/testPerformance.txt"

# Name of the maximum pulses count files
maxInputSpikesFile = paramDir + "/maxInputSpikes.npy"
maxOutputSpikesFile = paramDir + "/maxOutputSpikes.npy"

# Name of the active cycles counter file
cyclesCounterFile = paramDir + "/cyclesCounter.npy"
