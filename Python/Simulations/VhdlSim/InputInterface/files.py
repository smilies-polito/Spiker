mnistDir = "../../../../MNIST"

# Training
trainImages = mnistDir + "/train-images-idx3-ubyte"
trainLabels = mnistDir + "/train-labels-idx1-ubyte"

# Test
testImages = mnistDir + "/t10k-images-idx3-ubyte"
testLabels = mnistDir + "/t10k-labels-idx1-ubyte"

vhdlIoDir = "../../../../Vhdl/Hierarchical/Sim/IO"

vhdlAutoTestbench = "./vhdlAutoTestbench.sh"
vhdlInput = vhdlIoDir + "/vhdlImage.txt"
vhdlOutput = vhdlIoDir + "/vhdlSpikes.txt"
pythonOutput = vhdlIoDir + "/pythonSpikes.txt"
