import path_config
from vhdl_block import VHDLblock

class Shifter(VHDLblock):

	def __init__(self, bitwidth = 16, shift = 10):

		VHDLblock.__init__(self, entity_name = "shifter")

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
		self.entity.generic.add("N", "integer", str(bitwidth))

		if bitwidth <= 0:
			print("Invalid bit-width in shifter\n")
			exit(-1)

		elif shift < bitwidth and shift > 0:

			# Generics
			self.entity.generic.add("shift", "integer", 
					str(shift))


			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out(N-1 "
					"downto N-shift) <= (others => "
					"shifter_in(N-1));")
			self.architecture.bodyCodeHeader.add(
					"shifted_out(N-shift-1 "
					"downto 0) <= "
					"shifter_in(N-1 downto shift);")

		elif shift == bitwidth:

			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out <= "
					"(others =>"
					"shifter_in(shifter_in'length-1));")

		elif shift == 0:

			print("Zero")

			# Add/sub process
			self.architecture.bodyCodeHeader.add("shifted_out <= "
					"shifter_in;")

		else:
			print("Invalid shift value in shifter\n")
			exit(-1)
