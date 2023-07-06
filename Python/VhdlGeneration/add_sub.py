import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class AddSub(VHDLblock):

	def __init__(self, default_bitwidth = 8, output_dir = "output"):

		VHDLblock.__init__(self, entity_name = "add_sub")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(default_bitwidth))

		# Input ports
		self.entity.port.add("in0", "in", "signed(N-1 downto 0)")
		self.entity.port.add("in1", "in", "signed(N-1 downto 0)")
		self.entity.port.add("add_or_sub", "in", "std_logic")
		
		# Output ports
		self.entity.port.add("add_sub_out", "out", 
				"signed(N-1 downto 0)")

		# Add/sub process
		self.architecture.processes.add("operation")
		self.architecture.processes[0].sensitivity_list.add("in0")
		self.architecture.processes[0].sensitivity_list.add("in1")
		self.architecture.processes[0].sensitivity_list.add(
				"add_or_sub")
		self.architecture.processes[0].if_list.add()
		self.architecture.processes[0].if_list[0]._if_.conditions.add(
				"add_or_sub = '0'")
		self.architecture.processes[0].if_list[0]._if_.body.add(
				"add_sub_out <= in0 + in1;")
		self.architecture.processes[0].if_list[0]._else_.body.add(
				"add_sub_out <= in0 - in1;")


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
