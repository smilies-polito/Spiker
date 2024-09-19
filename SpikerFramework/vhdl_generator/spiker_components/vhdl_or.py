from .vhdl import sub_components, debug_component

from .vhdltools.vhdl_block import VHDLblock

class Or(VHDLblock):

	def __init__(self, bitwidth = 8, debug = False, debug_list = []):

		self.name = "generic_or"

		self.bitwidth = bitwidth
		self.components = sub_components(self)

		VHDLblock.__init__(self, entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		# Generics
		self.entity.generic.add("N", "integer", 
			str(self.bitwidth))

		# Input ports
		self.entity.port.add("or_in", "in", 
			"std_logic_vector(N-1 downto 0)")
		
		# Output ports
		self.entity.port.add("or_out", "out", 
				"std_logic")

		# Add/sub process
		self.architecture.processes.add("or_computation")
		self.architecture.processes["or_computation"].sensitivity_list.\
				add("or_in")
		self.architecture.processes["or_computation"].variables.add(name
				= "or_var", var_type = "std_logic")
		self.architecture.processes["or_computation"].bodyHeader.\
				add("or_var := '0';")
		self.architecture.processes["or_computation"].for_list.add(
				name = "or_loop", start = 0, 
				stop = "N-1", iter_name = "in_bit")
		self.architecture.processes["or_computation"].for_list[0].body.\
				add("or_var := or_var or or_in(in_bit);")
		self.architecture.processes["or_computation"].bodyFooter.\
				add("or_out <= or_var;")

		# Debug
		if debug:
			debug_component(self, debug_list)
