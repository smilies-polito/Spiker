from spikerplus.vhdl import AddSub

# Let's generate a simple adder/subtractor
adder = AddSub(

	bitwidth	= 8,

	# Saturate the output to max or min if the sum / subtraction overflows
	saturated	= False
)

print(adder.code())
