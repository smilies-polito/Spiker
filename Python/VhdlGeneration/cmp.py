import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class Cmp(VHDLblock):

	def __init__(self, bitwidth = 8, cmp_type = "gt", signal_type =
			"signed"):


		if signal_type != "std_logic" and signal_type != "signed" and \
			signal_type != "unsigned":

			print("Invalid register type")
			exit(-1)

		name = "cmp_" + cmp_type

		VHDLblock.__init__(self, entity_name = name)

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		if signal_type != "std_logic":
			self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(bitwidth))

		# Input ports
		if signal_type == "std_logic":
			self.entity.port.add("in0", "in", 
				"std_logic_vector(N-1 downto 0)")
			self.entity.port.add("in1", "in", 
				"std_logic_vector(N-1 downto 0)")

		else:
			self.entity.port.add("in0", "in", 
				signal_type + "(N-1 downto 0)")
			self.entity.port.add("in1", "in", 
				signal_type + "(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("cmp_out", "out", "std_logic")

		# Compare process
		self.architecture.processes.add("compare")
		self.architecture.processes["compare"].sensitivity_list.add(
				"in0")
		self.architecture.processes["compare"].sensitivity_list.add(
				"in1")

		self.architecture.processes["compare"].if_list.add()

		if cmp_type == "gt":
			self.architecture.processes["compare"].if_list[0]._if_.\
					conditions.add("in0 > in1")
		else:
			self.architecture.processes["compare"].if_list[0]._if_.\
					conditions.add("in0 = in1")


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
