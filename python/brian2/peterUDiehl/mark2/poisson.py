import brian2 as b2

def imgToSpikeTrain(network, image, inputIntensity):
	
	values = {
		"poissongroup":{
			"rates": image*b2.Hz/8*inputIntensity
		}
	}

	network.set_states(values, units=True, format='dict', level=0)

