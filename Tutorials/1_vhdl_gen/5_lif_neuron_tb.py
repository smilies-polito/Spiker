from spikerplus.vhdl import write_vhdl
from spikerplus.vhdl import LIFneuron_tb

# More complex component come with an already defined testbench
lif_tb = LIFneuron_tb(

	bitwidth 		= 16,

	# Recurrent weights (unclear name, it will be changed in next versions)
	w_inh_bw 		= 16,

	# Feed-forward weights
	w_exc_bw 		= 16,

	# Shift for the exponential computation
	shift 			= 8,

	# Reset to 0 when exceeding threshold
	reset			= "fixed",

	clock_period	= 20

)

write_vhdl(lif_tb)
