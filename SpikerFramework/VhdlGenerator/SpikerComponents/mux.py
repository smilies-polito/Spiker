from math import log2

from .vhdl import sub_components, debug_component
from .utils import ceil_pow2

from .vhdltools.vhdl_block import VHDLblock

class Mux(VHDLblock):

	def __init__(self, n_in = 8, in_type = "std_logic_vector",
			bitwidth = 4, debug = True, debug_list = []):

		# Name
		self.name = "mux_" + str(ceil_pow2(n_in)) + "to1"

		if in_type != "std_logic" and in_type != "std_logic_vector":
			self.name = self.name + "_" + in_type

		# Check input parameters
		if type(n_in) is not int or n_in < 2:
			raise ValueError("Invalid number of inputs in " +
					self.name)

		elif type(in_type) is not str or in_type != "std_logic" and \
		in_type != "std_logic_vector" and in_type != "signed" \
		and in_type != "unsigned":
			raise ValueError("Invalid input signal type in " + self.name)

		elif type(bitwidth) is not int or bitwidth < 1 or \
		bitwidth == 1 and in_type != "std_logic" or \
		bitwidth > 1 and in_type == "std_logic":
			raise ValueError("Invalid input bitwidth type in " +
					self.name)


		self.n_in = ceil_pow2(n_in)
		self.n_sel = int(log2(self.n_in))
		self.in_type = in_type
		self.bitwidth = bitwidth
		self.components = sub_components(self)

		VHDLblock.__init__(self, entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		if self.in_type != "std_logic" and self.in_type != "std_logic_vector":
			self.library["ieee"].package.add("numeric_std")

		if self.bitwidth > 1:
			self.entity.generic.add("bitwidth", "integer",
					str(self.bitwidth))

		# Input ports
		if self.n_sel == 1:
			sel_type = "std_logic"
			quote = "\'"
		else:
			sel_type = "std_logic_vector(" + str(self.n_sel-1) + \
					" downto 0)"
			quote = "\""

		self.entity.port.add("mux_sel", "in", sel_type)

		for port_number in range(self.n_in):

			port_name = "in" + str(port_number)

			if self.in_type == "std_logic":
				port_type = self.in_type
			else:
				port_type = self.in_type + "(bitwidth-1 downto 0)"

			self.entity.port.add(port_name, "in", port_type)
		
		# Output ports
		self.entity.port.add("mux_out", "out", port_type)

		# Selection process
		self.architecture.processes.add("selection")

		for key in self.entity.port:
			if self.entity.port[key].direction == "in":
				self.architecture.processes["selection"].\
					sensitivity_list.add(key)

		self.architecture.processes["selection"].case_list.\
			add("mux_sel")
		
		for port_number in range(self.n_in - 1):

			port_name = "in" + str(port_number)

			sel_value = quote + "{0:{fill}{width}{base}}".format(
					port_number, 
					fill = 0, 
					width = self.n_sel, 
					base = "b") + quote

			self.architecture.processes["selection"].\
				case_list["mux_sel"].when_list.add(sel_value)

			self.architecture.processes["selection"].\
				case_list["mux_sel"].when_list[sel_value].\
				body.add("mux_out <= " + port_name + ";")

		
		port_name = "in" + str(self.n_in - 1)

		self.architecture.processes["selection"].case_list["mux_sel"].\
			others.body.add("mux_out <= " + port_name + ";")


		# Debug
		if debug:
			debug_component(self, debug_list)
