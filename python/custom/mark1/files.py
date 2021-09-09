from utils import createParamDir

images = "../../brian2/mnist/train-images-idx3-ubyte"
labels = "../../brian2/mnist/train-labels-idx1-ubyte"

paramDir = "./parameters"

weightFilename = paramDir + "/weights"
thetaFilename = paramDir + "/theta"
performanceFilename = paramDir + "/performance"
assignmentsFilename = paramDir + "/assignments"
assignmentsFile = paramDir + "/assignments.npy"

createParamDir(paramDir)
