import subprocess as sp

from .vhdl import debug_component, sub_components

from .vhdltools.vhdl_block import VHDLblock

class Cmp(VHDLblock):

	def __init__(self, bitwidth = 8, cmp_type = "gt", signal_type =
			"signed", debug = False, debug_list = []):

		self.name = "cmp_" + cmp_type

		if signal_type != "std_logic" and signal_type != "signed" and \
			signal_type != "unsigned":
				raise ValueError("Invalid signal type in "
						+ self.name)

		self.bitwidth = bitwidth
		self.cmp_type = cmp_type
		self.signal_type = signal_type
		self.components = sub_components(self)

		VHDLblock.__init__(self, entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		if self.signal_type != "std_logic":
			self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(self.bitwidth))

		# Input ports
		if self.signal_type == "std_logic":
			self.entity.port.add("in0", "in", 
				"std_logic_vector(N-1 downto 0)")
			self.entity.port.add("in1", "in", 
				"std_logic_vector(N-1 downto 0)")

		else:
			self.entity.port.add("in0", "in", 
				self.signal_type + "(N-1 downto 0)")
			self.entity.port.add("in1", "in", 
				self.signal_type + "(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("cmp_out", "out", "std_logic")

		# Compare process
		self.architecture.processes.add("compare")
		self.architecture.processes["compare"].sensitivity_list.add(
				"in0")
		self.architecture.processes["compare"].sensitivity_list.add(
				"in1")

		self.architecture.processes["compare"].if_list.add()

		if self.cmp_type == "gt":
			self.architecture.processes["compare"].if_list[0]._if_.\
					conditions.add("in0 > in1")
		else:
			self.architecture.processes["compare"].if_list[0]._if_.\
					conditions.add("in0 = in1")


		self.architecture.processes["compare"].if_list[0]._if_.body.add(
				"cmp_out <= '1';")
		self.architecture.processes["compare"].if_list[0]._else_.body.\
				add("cmp_out <= '0';")

		# Debug
		if debug:
			debug_component(self, debug_list)
