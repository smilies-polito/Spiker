mnistDir = "../../../MNIST"

# Training
images = mnistDir + "/train-images-idx3-ubyte"
labels = mnistDir + "/train-labels-idx1-ubyte"

# Test
# images = mnistDir + "/t10k-images-idx3-ubyte"
# labels = mnistDir + "/t10k-labels-idx1-ubyte"

# Where to store the trained hyper-parameters and the time/accuracy results
paramDir = "./Parameters"

weightFilename = paramDir + "/weights"
thetaFilename = paramDir + "/theta"
performanceFilename = paramDir + "/performance"
assignementsFilename = paramDir + "/assignments"
assignementsFile = assignementsFilename + ".npy"
