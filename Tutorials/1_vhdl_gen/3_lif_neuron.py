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
	reset		= "fixed"
)

# Since there are many components within the neuron let's store the output on
# files
write_vhdl(lif)

# Now explore the files in the output directory. In particular the neuron
# component. You can customize the name of the directory passing output_dir =
# <name> to write_vhdl
