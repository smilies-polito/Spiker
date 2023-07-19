import subprocess as sp

import path_config

from vhdl_block import VHDLblock

class Mux4to1_signed(VHDLblock):

	def __init__(self, default_bitwidth = 8):

		VHDLblock.__init__(self, entity_name = "mux4to1_signed")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(default_bitwidth))

		# Input ports
		self.entity.port.add("sel", "in", 
				"std_logic_vector(1 downto 0)")
		self.entity.port.add("in0", "in", "signed(N-1 downto 0)")
		self.entity.port.add("in1", "in", "signed(N-1 downto 0)")
		self.entity.port.add("in2", "in", "signed(N-1 downto 0)")
		self.entity.port.add("in3", "in", "signed(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("mux_out", "out", "signed(N-1 downto 0)")

		# Add/sub process
		self.architecture.processes.add("selection")
		self.architecture.processes["selection"].sensitivity_list.add("sel")
		self.architecture.processes["selection"].sensitivity_list.add("in0")
		self.architecture.processes["selection"].sensitivity_list.add("in1")
		self.architecture.processes["selection"].sensitivity_list.add("in2")
		self.architecture.processes["selection"].sensitivity_list.add("in3")

		self.architecture.processes["selection"].case_list.add("sel")
		self.architecture.processes["selection"].case_list["sel"].when_list.add(
				'"00"')
		self.architecture.processes["selection"].case_list["sel"].when_list\
				['"00"'].body.add("mux_out <= in0;")
		self.architecture.processes["selection"].case_list["sel"].when_list.add(
				'"01"')
		self.architecture.processes["selection"].case_list["sel"].when_list\
				['"01"'].body.add("mux_out <= in1;")
		self.architecture.processes["selection"].case_list["sel"].when_list.add(
				'"10"')
		self.architecture.processes["selection"].case_list["sel"].when_list\
				['"10"'].body.add("mux_out <= in2;")

		self.architecture.processes["selection"].case_list["sel"].others.\
				body.add("mux_out <= in3;")


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
