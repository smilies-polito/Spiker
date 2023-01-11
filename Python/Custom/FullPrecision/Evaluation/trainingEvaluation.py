import sys
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Initialize the evaluation parameters
from runParameters import *

development = "../"

if development not in sys.path:
	sys.path.insert(1,development)


from createNetwork import createNetwork
from trainFunctions import run




# Create the network
network = createNetwork(networkList, None, None, mode, 
			excDictList, scaleFactors, inh2excWeights)


# Create a spare matrix of random spikes
spikesTrains = sp.random(N_sim, networkList[0], density = density)
spikesTrains = spikesTrains.A.astype(bool)


counter = run(network, networkList, spikesTrains, dt_tauDict, stdpDict, mode,
		constSums)

print("Counter: ", counter)
