from .vhdl import sub_components, debug_component

from .vhdltools.vhdl_block import VHDLblock

class AddrConverter(VHDLblock):

	def __init__(self, bitwidth = 8, debug = False, debug_list = []):

		self.name = "addr_converter"
		self.bitwidth = bitwidth
		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)

	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(self.bitwidth))

		# Input ports
		self.entity.port.add("addr_in", "in", "std_logic_vector(N-1 "
				"downto 0)")
		
		# Output ports
		self.entity.port.add("addr_out", "out", "std_logic_vector(N-1 "
				"downto 0)")

		# Add/sub process
		self.architecture.bodyCodeHeader.add(
			"addr_out <= std_logic_vector(unsigned(addr_in) + "
			"to_unsigned(1, N));"
		)

		# Debug
		if debug:
			debug_component(self, debug_list)
