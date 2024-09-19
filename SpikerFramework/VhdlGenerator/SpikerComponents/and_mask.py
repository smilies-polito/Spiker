from .vhdl import sub_components, debug_component

from .vhdltools.vhdl_block import VHDLblock

class AndMask(VHDLblock):

	def __init__(self, data_type = "std_logic_vector", debug = False,
			debug_list = []):

		self.name = "and_mask"
		self.data_type = data_type
		self.components = sub_components(self)

		VHDLblock.__init__(self, self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")


		self.entity.generic.add(
			name 		= "N",
			gen_type	= "integer",
			value 		= 8
		)

		self.entity.port.add(
			name 		= "input_bits",
			direction 	= "in",
			port_type	= self.data_type + "(N-1 downto 0)"
		)

		self.entity.port.add(
			name 		= "mask_bit",
			direction 	= "in",
			port_type	= "std_logic"
		)

		self.entity.port.add(
			name 		= "output_bits",
			direction 	= "out",
			port_type	= self.data_type + "(N-1 downto 0)"
		)

		self.architecture.processes.add("mask")
		self.architecture.processes["mask"].sensitivity_list.add(
				"input_bits")
		self.architecture.processes["mask"].sensitivity_list.add(
				"mask_bit")

		self.architecture.processes["mask"].for_list.add(
			start 		= 0,
			stop 		= "N-1"
		)
		self.architecture.processes["mask"].for_list[0].body.add(
			"output_bits(i) <= input_bits(i) and mask_bit;")

		# Debug
		if debug:
			debug_component(self, debug_list)
