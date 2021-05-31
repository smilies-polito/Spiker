#!/Users/alessio/anaconda3/bin/python3

from layer import updateLayer

# Update the entire neural network.
#
# INPUT PARAMETERS:
#
# 	1) inEvents: boolean NumPy array containing the input of the network in
# 	the current time step
#
# 	2) networkDictList: list of dictionaries. Each of them contains the
# 	parameters of a specific layer
#
# 	3) dt_tau: numerical value corresponding to the ratio between the time
# 	step and the exponential time constant. See updateLayer for more
# 	details.
#
#	4) v_reset: voltage at which the membrane potential of each neuron is
#	reset in case it exceeds the threshold.

def updateNetwork(inEvents, networkDictList, dt_tau, v_reset):

	# Update the first layer with the input spikes
	updateLayer(inEvents, networkDictList[0], dt_tau, v_reset)

	for i in range(1,len(networkDictList)):

		# Propagate the spikes along the network
		updateLayer(networkDictList[i-1]["outEvents"],
			networkDictList[i], dt_tau, v_reset) 
