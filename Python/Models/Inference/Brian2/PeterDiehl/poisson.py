import brian2 as b2

def imgToSpikeTrain(network, image, inputIntensity):

	"""
	Associate the inputs of the Poisson layer to the values of the input
	image.

	INPUT:

		1) network: Brian 2 Network object.

		2) image: Numpy array of float values.

		3) inputIntensity: float. Variable normalization factor. If the
		spiking activity is too low the input intensity can be
		increased.
	"""
	
	# Normalize input values and treat them as average spiking frequencies
	values = {
		"poissongroup":{
			"rates": image*b2.Hz/8*inputIntensity
		}
	}

	# Set the frequencies in the input layer
	network.set_states(values, units=True, format='dict', level=0)
