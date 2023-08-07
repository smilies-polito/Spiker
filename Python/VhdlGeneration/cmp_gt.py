import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class CmpGt(VHDLblock):

	def __init__(self, default_bitwidth = 8):

		VHDLblock.__init__(self, entity_name = "cmp_gt")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(default_bitwidth))

		# Input ports
		self.entity.port.add("in0", "in", "signed(N-1 downto 0)")
		self.entity.port.add("in1", "in", "signed(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("cmp_out", "out", "std_logic")

		# Compare process
		self.architecture.processes.add("compare")
		self.architecture.processes["compare"].sensitivity_list.add(
				"in0")
		self.architecture.processes["compare"].sensitivity_list.add(
				"in1")

		self.architecture.processes["compare"].if_list.add()
		self.architecture.processes["compare"].if_list[0]._if_.\
				conditions.add("in0 > in1")

		self.architecture.processes["compare"].if_list[0]._if_.body.add(
				"cmp_out <= '1';")
		self.architecture.processes["compare"].if_list[0]._else_.body.\
				add("cmp_out <= '0';")


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
