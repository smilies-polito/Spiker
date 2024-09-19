from .vhdl import sub_components, debug_component

from .vhdltools.vhdl_block import VHDLblock

class Shifter(VHDLblock):

	def __init__(self, bitwidth = 16, shift = 10, debug = True, 
			debug_list = []):

		self.name = "shifter"

		self.bitwidth = bitwidth
		self.shift = shift
		self.components = sub_components(self)

		VHDLblock.__init__(self, entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Input ports
		self.entity.port.add("shifter_in", "in", "signed(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("shifted_out", "out", 
				"signed(N-1 downto 0)")

		# Generics
		self.entity.generic.add("N", "integer", str(self.bitwidth))

		if self.bitwidth <= 0:
			raise ValueError("Invalid bit-width in " + self.name)

		elif self.shift < self.bitwidth and self.shift > 0:

			# Generics
			self.entity.generic.add("shift", "integer", 
					str(self.shift))


			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out(N-1 "
					"downto N-shift) <= (others => "
					"shifter_in(N-1));")
			self.architecture.bodyCodeHeader.add(
					"shifted_out(N-shift-1 "
					"downto 0) <= "
					"shifter_in(N-1 downto shift);")

		elif self.shift >= self.bitwidth:

			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out <= "
					"(others =>"
					"shifter_in(shifter_in'length-1));")

		elif self.shift == 0:

			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out <= "
					"shifter_in;")

		else:
			raise ValueError("Invalid shift value in " + self.name)

		# Debug
		if debug:
			debug_component(self, debug_list)
