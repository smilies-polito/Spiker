import subprocess as sp

import path_config

from vhdl_block import VHDLblock
from if_statement import If

class RegSignedSyncRst(VHDLblock):

	def __init__(self, default_bitwidth = 8):

		VHDLblock.__init__(self, entity_name = "reg_signed_sync_rst")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(default_bitwidth))

		# Input ports
		self.entity.port.add("clk", "in", "std_logic")
		self.entity.port.add("en", "in", "std_logic")
		self.entity.port.add("rst_n", "in", "std_logic")
		self.entity.port.add("reg_in", "in", "signed(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("reg_out", "out", "signed(N-1 downto 0)")

		# Sample process
		self.architecture.processes.add("sample")
		self.architecture.processes[0].sensitivity_list.add("clk")
		self.architecture.processes[0].sensitivity_list.add("en")

		# Reset inner if statement
		reset_if = If()

		reset_if._if_.conditions.add("rst_n = '0'")
		reset_if._if_.body.add("reg_out <= (others => '0');")
		reset_if._elsif_.add()
		reset_if._elsif_[0].conditions.add("en = '1'")
		reset_if._elsif_[0].body.add("reg_out <= reg_in;")

		# Clk outer if statement
		self.architecture.processes[0].if_list.add()
		self.architecture.processes[0].if_list[0]._if_.conditions.add(
				"clk'event")
		self.architecture.processes[0].if_list[0]._if_.conditions.add(
				"clk = '1'", "and")

		self.architecture.processes[0].if_list[0]._if_.body.add(
				reset_if)



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
