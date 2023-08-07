import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class Shifter(VHDLblock):

	def __init__(self, default_bitwidth = 16, default_shift = 10):

		VHDLblock.__init__(self, entity_name = "shifter")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Input ports
		self.entity.port.add("shifter_in", "in", "signed(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("shifted_out", "out", 
				"signed(N-1 downto 0)")

		# Generics
		self.entity.generic.add("N", "integer", str(default_bitwidth))

		if default_bitwidth <= 0:
			print("Invalid bit-width in shifter\n")
			exit(-1)

		elif default_shift < default_bitwidth and default_shift > 0:

			# Generics
			self.entity.generic.add("shift", "integer", 
					str(default_shift))


			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out(N-1 "
					"downto N-shift) <= (others => "
					"shifter_in(N-1));")
			self.architecture.bodyCodeHeader.add(
					"shifted_out(N-shift-1 "
					"downto 0) <= "
					"shifter_in(N-1 downto shift);")

		elif default_shift == default_bitwidth:

			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out <= "
					"(others =>"
					"shifter_in(shifter_in'length-1));")

		elif default_shift == 0:

			print("Zero")

			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out <= "
					"shifter_in;")

		else:
			print("Invalid shift value in shifter\n")
			exit(-1)


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
