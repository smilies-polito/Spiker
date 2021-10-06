import numpy as np

from poisson import *

from equationsParameters import *
from neuronsParameters import *

# Add the parameters of the network to the local variables
locals().update(parametersDict)




def run(network, trainDuration, currentSpikesCount, prevSpikesCount):

	'''
	Train the network over the pixels spikes trains.

	INPUT:

		1) network: Brian2 Network object containing the complete
		network structure.

		2) trainDuration: time duration of the spikes train expressed in
		milleseconds.


		3) currentSpikesCount: NumPy array with size equal to the number
		of elements in the last layer. 

		4) prevSpikesCount: NumPy array containing the count of the
		spikes generated by the network up to the previous training
		cycle.
	'''

	# Train the network over the pixels' spikes train
	network.run(trainDuration)
	
	# Update the count of the spikes generated by the network
	updatePulsesCount(network, currentSpikesCount, prevSpikesCount)







def updatePulsesCount(network, currentSpikesCount, prevSpikesCount):

	'''
	Update the count of the spikes generated along a training cycle over a
	complete train of spikes.

	INPUT:

		1) network: Brian2 Network object containing the complete
		network structure.

		2) currentSpikesCount: NumPy array with size equal to the number
		of elements in the last layer. 

		3) prevSpikesCount: NumPy array containing the count of the
		spikes generated by the network up to the previous training
		cycle.

	'''

	# Get the total spikes' count from the beginnning up to now.
	spikeMonitorCount = network.get_states(
				units=True, 
				format='dict', 
				subexpressions=False, 
				read_only_variables=True,
				level=0)["spikemonitor"]["count"]

	# Compute the spikes' count relative to the current image
	currentSpikesCount[:] = np.asarray(spikeMonitorCount) - prevSpikesCount
	
	# Update the total count
	prevSpikesCount[:] = np.asarray(spikeMonitorCount)






def rest(network, restTime, imageSize):

	'''
	Bring the network into a rest state.

	INPUT:

		1) network: Brian2 Network object containing the complete
		network structure.

		2) restTime: time duration of the resting period expressed
		in milliseconds.

		3) imageSize: total number of pixels composing the image.

	'''

	# Reset to zero the spikes trains
	imgToSpikeTrain(network, np.zeros(imageSize), 0)

	# Run the network for a resting period 
	network.run(restTime)
