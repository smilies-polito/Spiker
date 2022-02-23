import numpy as np

def debugNetwork(network, networkList, rootLayerName, rootExcSynapseName,
		rootInhSynapseName, inputCounter, spikesCounter, trainDuration,
		currentIndex, label):

	N_layers = len(networkList)

	spikesCounter.insert(0, inputCounter)
	
	for i in range(1, N_layers):

		layerName = rootLayerName + str(i)
		excSynapseName = rootExcSynapseName + str(i)
		inhSynapseName = rootInhSynapseName + str(i)

		print("Layer " + str(i))

		print("Rest potential: ", network[layerName]["vRest"])
		print("Reset potential: ", network[layerName]["vReset"])
		print("Threshold: ", network[layerName]["vThresh"])

		print("Weights sum")
		print(np.sum(network[excSynapseName]["weights"], axis = 1))

		print("Weights average")
		print(np.average(network[excSynapseName]["weights"], axis = 1))

		print("Weights max")
		print(np.max(network[excSynapseName]["weights"], axis = 1))

		print("Weights min")
		print(np.min(network[excSynapseName]["weights"], axis = 1))

		print("Inhibitory weight")
		print(network[inhSynapseName]["weight"])

		print("Input pulses count")
		print(spikesCounter[i-1])

		print("Input pulses average")
		print(spikesCounter[i-1]/trainDuration)

		print("Output pulses count")
		print(spikesCounter[i])

		print("Output pulses average")
		print(spikesCounter[i]/trainDuration)

		print("Image number: ", currentIndex)
		print("Label: ", label)


		print("\n\n")
