import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class GenericMux_1bit(VHDLblock):

	def __init__(self, default_sel_bitwidth = 8):

		VHDLblock.__init__(self, entity_name = "generic_mux_1bit")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N_sel", "integer", 
			str(default_sel_bitwidth))

		# Input ports
		self.entity.port.add("mux_in", "in", 
			"std_logic_vector(2**N_sel-1 downto 0)")
		self.entity.port.add("mux_sel", "in", 
			"std_logic_vector(N_sel-1 downto 0)")
		
		# Output ports
		self.entity.port.add("mux_out", "out", 
				"std_logic")

		# Add/sub process
		self.architecture.processes.add("selection")
		self.architecture.processes["selection"].sensitivity_list.\
				add("mux_in")
		self.architecture.processes["selection"].sensitivity_list.\
				add("mux_sel")
		self.architecture.processes["selection"].bodyHeader.add(
				"mux_out <= mux_in(to_integer("
				"unsigned(mux_sel)));")


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
