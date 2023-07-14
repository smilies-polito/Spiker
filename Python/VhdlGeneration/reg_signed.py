import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class RegSigned(VHDLblock):

	def __init__(self, default_bitwidth = 8):

		VHDLblock.__init__(self, entity_name = "reg_signed")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(default_bitwidth))

		# Input ports
		self.entity.port.add("clk", "in", "std_logic")
		self.entity.port.add("en", "in", "std_logic")
		self.entity.port.add("reg_in", "in", "signed(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("reg_out", "out", "signed(N-1 downto 0)")

		# Add/sub process
		self.architecture.processes.add("sample")
		self.architecture.processes["sample"].sensitivity_list.add("clk")
		self.architecture.processes["sample"].sensitivity_list.add("en")

		self.architecture.processes["sample"].if_list.add()
		self.architecture.processes["sample"].if_list[0]._if_.conditions.add(
				"clk'event")
		self.architecture.processes["sample"].if_list[0]._if_.conditions.add(
				"clk = '1'", "and")
		self.architecture.processes["sample"].if_list[0]._if_.conditions.add(
				"en = '1'", "and")

		self.architecture.processes["sample"].if_list[0]._if_.body.add(
				"reg_out <= reg_in;")


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
