from utils import createDir

# Complete path of the MNIST files
images = "../../brian2/mnist/train-images-idx3-ubyte"
labels = "../../brian2/mnist/train-labels-idx1-ubyte"

# Directory in which parameters and performance of the network are stored
paramDir = "./parameters"
createDir(paramDir)

# Name of the parameters files
weightFilename = paramDir + "/weights"
thetaFilename = paramDir + "/theta"
assignmentsFile = paramDir + "/assignments.npy"

# Name of the performance files
performanceFile = paramDir + "/performance.txt"
