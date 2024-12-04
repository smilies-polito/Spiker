from spikerplus.vhdl import AddSub, Testbench

# Let's generate a simple adder/subtractor
adder = AddSub(

	bitwidth	= 8,

	# Saturate the output to max or min if the sum / subtraction overflows
	saturated	= False
)

# And now generate a testbench skeleton which we can fill later
adder_tb = Testbench(
	dut				= adder,
	clock_period	= 20
)

print(adder_tb.code())
