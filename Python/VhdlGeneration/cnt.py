import subprocess as sp

import path_config

from if_statement import If
from vhdl_block import VHDLblock

class Cnt(VHDLblock):

	def __init__(self, default_bitwidth = 8):

		VHDLblock.__init__(self, entity_name = "cnt")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(default_bitwidth))

		# Input ports
		self.entity.port.add("clk", "in", "std_logic")
		self.entity.port.add("cnt_en", "in", "std_logic")
		self.entity.port.add("cnt_rst_n", "in", "std_logic")
		
		# Output ports
		self.entity.port.add("cnt_out", "out", "std_logic_vector(N-1 "
			"downto 0)")

		# Add/sub process
		self.architecture.processes.add("count")
		self.architecture.processes["count"].sensitivity_list.add(
				"clk")
		self.architecture.processes["count"].variables.add(
				name = "cnt_var",
				var_type = "integer",
				value = "0"
		)

		en_if = If()
		en_if._if_.conditions.add("cnt_en = \'1\'")
		en_if._if_.body.add("cnt_var := cnt_var + 1;")

		rst_if = If()
		rst_if._if_.conditions.add("cnt_rst_n = \'0\'")
		rst_if._if_.body.add("cnt_var := 0;")
		rst_if._else_.body.add(en_if)

		self.architecture.processes["count"].if_list.add()
		self.architecture.processes["count"].if_list[0]._if_.\
				conditions.add("clk'event")
		self.architecture.processes["count"].if_list[0]._if_.\
				conditions.add("clk = '1'", "and")

		self.architecture.processes["count"].if_list[0]._if_.body.add(
				rst_if)

		self.architecture.processes["count"].bodyFooter.add(
				"cnt_out <= std_logic_vector(to_unsigned("
				"cnt_var, N));")


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
