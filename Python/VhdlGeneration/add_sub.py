from vhdl import sub_components, debug_component

import path_config
from vhdl_block import VHDLblock

class AddSub(VHDLblock):

	def __init__(self, bitwidth = 8, debug = False, debug_list = []):

		self.name = "add_sub"
		self.bitwidth = bitwidth
		self.components = sub_components(self)

		VHDLblock.__init__(self, entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(self.bitwidth))

		# Input ports
		self.entity.port.add("in0", "in", "signed(N-1 downto 0)")
		self.entity.port.add("in1", "in", "signed(N-1 downto 0)")
		self.entity.port.add("add_or_sub", "in", "std_logic")
		
		# Output ports
		self.entity.port.add("add_sub_out", "out", 
				"signed(N-1 downto 0)")

		# Add/sub process
		self.architecture.processes.add("operation")
		self.architecture.processes["operation"].sensitivity_list.add(
				"in0")
		self.architecture.processes["operation"].sensitivity_list.add(
				"in1")
		self.architecture.processes["operation"].sensitivity_list.add(
				"add_or_sub")
		self.architecture.processes["operation"].if_list.add()
		self.architecture.processes["operation"].if_list[0]._if_.\
				conditions.add("add_or_sub = '0'")
		self.architecture.processes["operation"].if_list[0]._if_.body.\
				add("add_sub_out <= in0 + in1;")
		self.architecture.processes["operation"].if_list[0]._else_.\
				body.add("add_sub_out <= in0 - in1;")

		# Debug
		if debug:
			debug_component(self, debug_list)
