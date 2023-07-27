import subprocess as sp
import path_config
from vhdl_block import VHDLblock

class AndMask(VHDLblock):

	def __init__(self, data_type = "std_logic_vector"):

		VHDLblock.__init__(self, "and_mask")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")


		self.entity.generic.add(
			name 		= "N",
			gen_type	= "integer",
			value 		= 8
		)

		self.entity.port.add(
			name 		= "input_bits",
			direction 	= "in",
			port_type	= data_type + "(N-1 downto 0)"
		)

		self.entity.port.add(
			name 		= "mask_bit",
			direction 	= "in",
			port_type	= "std_logic"
		)

		self.entity.port.add(
			name 		= "output_bits",
			direction 	= "out",
			port_type	= data_type + "(N-1 downto 0)"
		)

		self.architecture.processes.add("mask")
		self.architecture.processes["mask"].sensitivity_list.add(
				"input_bits")
		self.architecture.processes["mask"].sensitivity_list.add(
				"mask_bit")

		self.architecture.processes["mask"].for_list.add(
			start 		= 0,
			stop 		= "N-1"
		)
		self.architecture.processes["mask"].for_list[0].body.add(
			"output_bits(i) <= input_bits(i) and mask_bit;")

	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")



	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")
