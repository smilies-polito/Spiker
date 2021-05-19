#!/Users/alessio/anaconda3/bin/python3

# Script which simulates the model developed with an image extracted from the mnist

import numpy as np
from time import perf_counter

from snnDataStruct import createNetworkDictList
from mnist import loadDataset
from poisson import randInt3Dgen, poisson
from training import trainSingleImg


images = "../mnist/train-images-idx3-ubyte"
labels = "../mnist/train-labels-idx1-ubyte"

# Number of simulation cycles
N_sim = 100

# Number of pixels pf each image
N_pixels = 784

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

dt_tau = 0.005

A_ltp = 0.001
A_ltd = -0.001


# Array containing the number classification
labelsArray = np.linspace(0, networkList[-1]-1, networkList[-1]).astype(int)

print("Initial labels array")
print(labelsArray)
print("\n")

# Array that will contain the count of the output spikes for each neuron
spikeCountArray = np.zeros(networkList[-1]).astype(int)

# Create the network dictionary list
networkDictList = createNetworkDictList(v_th_list, v_reset, w_min_list, w_max_list, 
			networkList)


# Load all the images and labels from the dataset
imgArray, labels = loadDataset(images, labels)


random2D = randInt3Dgen(1, N_sim, N_pixels, 0, 255)[0]
poissonImg = poisson(imgArray[0], random2D)


# Starting time of the training
start = perf_counter()

# Train the network
accuracy = trainSingleImg(poissonImg, labels[0], labelsArray, networkDictList, dt_tau, 
				v_reset, A_ltp, A_ltd, spikeCountArray)

# Ending time of the training
stop = perf_counter()

print("\nSpike Count Array")
print(spikeCountArray)
print("\n")

print("\nClassification result")
print(accuracy)
print("\n")

print("\nFinal labels array")
print(labelsArray)
print("\n")

print("\nTraining time")
trainingTime = str(stop-start) + " seconds"
print(trainingTime)
print("\n")





