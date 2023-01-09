mnistDir = "../../../MNIST"

# Training
images = mnistDir + "/train-images-idx3-ubyte"
labels = mnistDir + "/train-labels-idx1-ubyte"

# Test
images = mnistDir + "/t10k-images-idx3-ubyte"
labels = mnistDir + "/t10k-labels-idx1-ubyte"

# Where to store the trained hyper-parameters and the time/accuracy results
paramDir = "./parameters"

weightFilename = paramDir + "/weights"
thetaFilename = paramDir + "/theta"
performanceFilename = paramDir + "/performance"
assignementsFilename = paramDir + "/assignements"
assignementsFile = assignementsFilename + ".npy"
