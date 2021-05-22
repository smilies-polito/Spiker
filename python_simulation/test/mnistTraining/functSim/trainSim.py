#!/Users/alessio/anaconda3/bin/python3

# Script which simulates the model developed with an image extracted from the mnist

import sys

previousDir = ".."

if previousDir not in sys.path:
	sys.path.insert(1, previousDir)

import numpy as np
from time import perf_counter

from snnDataStruct import createNetworkDictList
from mnist import loadDataset
from training import train


images = "../../mnist/train-images-idx3-ubyte"
labels = "../../mnist/train-labels-idx1-ubyte"


# Number of simulation cycles
timeEvolCycles = 100

# Number of pixels pf each image
N_pixels = 784
pixelMin = 0
pixelMax = 255

# List describing the network. Each entry corresponds to the number of nodes in the
# specific layer
networkList = [N_pixels, 500, 10]


# Thresholds
v_th_list = [250*np.ones(networkList[i]) for i in range(1,len(networkList))]

# Membrane potential reset state
v_reset = 0 

# Range of the weights to generate
w_min_list = [0*np.ones(networkList[i]) for i in range(1,len(networkList))]
w_max_list = [1*np.ones(networkList[i]) for i in range(1,len(networkList))]

v_mem_dt_tau = 0.1
stdp_dt_tau = 0.03

A_ltp = 0.001
A_ltd = -0.001


# Array containing the number classification
labelsArray = np.linspace(0, networkList[-1]-1, networkList[-1]).astype(int)

# Array that will contain the count of the output spikes for each neuron
classificationArray = np.zeros((networkList[-1], networkList[-1])).astype(int)

# Create the network dictionary list
networkDictList = createNetworkDictList(v_th_list, v_reset, w_min_list, w_max_list, 
			networkList)


# Load all the images and labels from the dataset
imgages, labels = loadDataset(images, labels)

N_subsets = 100

# Starting time of the training
start = perf_counter()


train(imgages, labels, N_subsets, timeEvolCycles, N_pixels, pixelMin, pixelMax, 
		labelsArray, networkDictList, v_mem_dt_tau, stdp_dt_tau, v_reset, A_ltp, 
		A_ltd, classificationArray)


# Ending time of the training
stop = perf_counter()

print("\nTraining time")
trainingTime = str(stop-start) + " seconds"
print(trainingTime)
print("\n")

