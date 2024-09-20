from .vhdl import sub_components, debug_component

from .vhdltools.vhdl_block import VHDLblock

class Multiplier(VHDLblock):

	def __init__(self, bitwidth = 8, debug = True, debug_list = []):

		self.name = "mult_signed"
		self.bitwidth = 8
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
		
		# Output ports
		self.entity.port.add("mult_out", "out", "signed(N-1 downto 0)")

		# Multiplication
		self.architecture.bodyCodeHeader.add("mult_out <= in0 * in1;")

		# Debug
		if debug:
			debug_component(self, debug_list)
