from spikerplus.vhdl import write_vhdl
from spikerplus.vhdl import LIFneuron

# Generate a more complex block: a Leaky Integrate and Fire neuron
lif = LIFneuron(

	bitwidth 	= 16,

	# Recurrent weights (unclear name, it will be changed in next versions)
	w_inh_bw 	= 6,

	# Feed-forward weights
	w_exc_bw 	= 6,

	# Shift for the exponential computation
	shift 		= 4,

	# Reset to 0 when exceeding threshold
	reset		= "fixed",

	# You will be prompted to decide which signals you want to bring outside
	debug		= True
)

# Since there are many components within the neuron let's store the output on
# files
write_vhdl(lif)

# Now explore the files in the output directory. In particular the neuron
# component. You will now see the signals you selected in the entity of the
# neuron, so they are exposed for you to analyze them. The name will be composed
# as 
#
# 	name of the component in which the signal is declared + signal name
