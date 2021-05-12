Directory organization.

The directories follow the project flow that has led to the optimized version of the
spiking neural network software model:


	1) singleCycle: first model developed. It contains the description of a spiking
	   neuron with a single input and modifies the membrane potential and the output 
	   spikes integrating the spikes coming from this usique input.


	2) multiCycle: it contains the models of a neuron, a layer of neurons and a
	   complete network. The neuron is updated looping over all the inputs, the
	   layer is updated looping over all the neurons and the network is updated 
	   looping over all the layers. Quite unoptimized due to the nested loops.

	3) testWithLoop: it contains the models of a neuron, a layer of neurons and a
	   complete network. In this case the neuron is updated adding together the
	   active inputs in parallel. The layer and the network are still updated through
	   a loop. The model considered is still the same but more optimized at the
	   neuron level. The "test" word in the name underlines the fact that the network
	   is still not ready to be trained, it can only evolve using predetermined
	   weights.

	4) parallelTest: it contains the models of a neuron, a layer of neurons and a
	   complete network. Here both the neuron and the layer are updated in parallel,
	   without any loop. This allows what is the higher possible level of 
	   optimization. This is because the network data structure is composed by layers 
	   that are different in size and so it can't be implemented using a 
	   tridimensional NumPy array, but requires a list. This list can only be updated
	   through a loop. This is not critical in general because the amount of layers
	   that compose the network is usually limited to some units.

	5) parallelTraining: it contains the models of a neuron, a layer of neurons and a
	   complete network. The implementation is the same used in parallelTest but in
	   this case the network is ready to be trained.
