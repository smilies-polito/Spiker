import brian2 as b2


def imgToSpikeTrain(network, image, inputIntensity):
	
	""" 
	Convert a black and white image into spike trains using the Poisson
	method.

	INPUT:

		1) network: Brian2 Network object containing the complete
		network structure.

		2) image: NumPy array containing the value of each pixel
		expressed as an integer.

		3) inputIntensity: current value of the pixel"s intensity.

	"""

	# Interpret each pixel"s value as a frequency value
	values = {
		"poissongroup":{
			"rates": image*b2.Hz/8*inputIntensity
		}
	}

	# Set the rates of the Poisson layer
	network.set_states(values, units=True, format="dict", level=0)
