import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class Or(VHDLblock):

	def __init__(self, bitwidth = 8):

		VHDLblock.__init__(self, entity_name = "generic_or")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		# Generics
		self.entity.generic.add("N", "integer", 
			str(bitwidth))

		# Input ports
		self.entity.port.add("or_in", "in", 
			"std_logic_vector(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("or_out", "out", 
				"std_logic")

		# Add/sub process
		self.architecture.processes.add("or_computation")
		self.architecture.processes["or_computation"].sensitivity_list.\
				add("or_in")
		self.architecture.processes["or_computation"].variables.add(name
				= "or_var", var_type = "std_logic")
		self.architecture.processes["or_computation"].bodyHeader.\
				add("or_var := '0';")
		self.architecture.processes["or_computation"].for_list.add(
				name = "or_loop", start = 0, 
				stop = "N-1", iter_name = "in_bit")
		self.architecture.processes["or_computation"].for_list[0].body.\
				add("or_var := or_var or or_in(in_bit);")
		self.architecture.processes["or_computation"].bodyFooter.\
				add("or_out <= or_var;")


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
