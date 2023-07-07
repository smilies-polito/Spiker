import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class Shifter(VHDLblock):

	def __init__(self, default_bitwidth = 16, default_shift = 10, 
			output_dir = "output"):

		VHDLblock.__init__(self, entity_name = "shifter")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(default_bitwidth))
		self.entity.generic.add("shift", "integer", str(default_shift))

		# Input ports
		self.entity.port.add("shifter_in", "in", "signed(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("shifted_out", "out", 
				"signed(N-1 downto 0)")

		# Add/sub process
		self.architecture.bodyCodeHeader.add("shifted_out(N-1 "
				"downto N-shift) <= (others => "
				"shifter_in(N-1));")
		self.architecture.bodyCodeHeader.add("shifted_out(N-shift-1 "
				"downto 0) <= shifter_in(N-1 downto shift);")
		self.write_file(output_dir)


	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl " + self.entity.name + ".vhd" + "; "

		sp.run(command, shell = True)

		print("\n")
	

	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")

bit_width = 16

for shift in range(0, bit_width+1):

	print("shift = %d" %(shift))

	shifter = Shifter(default_bitwidth = bit_width, default_shift = shift)

	shifter.compile()
	shifter.elaborate()
