Directory organization.

The various directories contain the scripts used to simulate the models stored in 
../development. In particular:

	1) singleCycle and multiCycle: contain a first version of the simulation scripts
	   that graphically represent the temporal evolution of the input spikes,
	   membrane potentials and output spikes.

	2) testWithLoop: updates the previous simulation scripts using a more gerarchical
	   organization, with high level functions that carry out the various
	   simulation tasks. In the simulation of the neuron and the layer the plots are
	   simply visualized on the screen, while for the network they are stored in the
	   directory plots. This is because there can be many plots and storing them
	   can help in visualizing and comparing them later.

	3) parallelTest: adapts the functions and scripts of testWithLoop to fit the new
	   function prototypes. In this case the complete network plots are visualized
	   on the screen as in the case of the neuron and the layer, without storing them
	   in a separated directory.

	4) parallelTraining: this still contains scripts that allow a graphical
	   simulation of the network, plotting input and output spikes, membrane
	   potentials and the temporal evolution of the weights, in order to be able to
	   compared it with the spikes arrival time.

	5) plots: contains the plots stored by testWithLoop/snn_sim.py
