from .vhdl import sub_components, debug_component

from .vhdltools.vhdl_block import VHDLblock

class Decoder(VHDLblock):

	def __init__(self, bitwidth = 8, debug = False, debug_list = []):

		self.name	= "decoder"
		self.bitwidth	= bitwidth
		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("bitwidth", "integer", str(self.bitwidth))

		# Input ports
		self.entity.port.add("encoded_in", "in", "std_logic_vector("
			"bitwidth-1 downto 0)")
		
		# Output ports
		self.entity.port.add("decoded_out", "out", "std_logic_vector("
			"2**bitwidth-1 downto 0)")

		# Decode process
		self.architecture.processes.add("decode")
		self.architecture.processes["decode"].sensitivity_list.add(
				"encoded_in")
		self.architecture.processes["decode"].bodyHeader.add(
			"decoded_out <= (others => \'0\');"
		)
		self.architecture.processes["decode"].bodyHeader.add(
				"decoded_out(to_integer(unsigned(encoded_in)))"
				" <= \'1\';"
		)

		# Debug
		if debug:
			debug_component(self, debug_list)
