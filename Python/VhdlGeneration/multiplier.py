import path_config
from vhdl_block import VHDLblock

class Multiplier(VHDLblock):

	def __init__(self, bitwidth = 8):

		VHDLblock.__init__(self, entity_name = "mult_signed")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(bitwidth))

		# Input ports
		self.entity.port.add("in0", "in", "signed(N-1 downto 0)")
		self.entity.port.add("in1", "in", "signed(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("mult_out", "out", "signed(N-1 downto 0)")

		# Multiplication
		self.architecture.bodyCodeHeader.add("mult_out <= in0 * in1;")
