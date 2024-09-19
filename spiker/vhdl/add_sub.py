from .vhdl import sub_components, debug_component

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If

class AddSub(VHDLblock):

	def __init__(self, bitwidth = 8, saturated = False, debug = False,
			debug_list = []):

		self.name	= "add_sub"
		self.bitwidth	= bitwidth
		self.saturated	= saturated
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
		self.entity.port.add("add_or_sub", "in", "std_logic")
		
		# Output ports
		self.entity.port.add("add_sub_out", "out", 
				"signed(N-1 downto 0)")

		if self.saturated:

			self.architecture.constant.add(
				name		= "sat_up",
				const_type	= "integer",
				value		= "2**(N-1)-1"
			)

			self.architecture.constant.add(
				name		= "sat_down",
				const_type	= "integer",
				value		= "-2**(N-1)"
			)

			self.architecture.signal.add(
				name		= "local_in0",
				signal_type	= "signed(N downto 0)"
			)

			self.architecture.signal.add(
				name		= "local_in1",
				signal_type	= "signed(N downto 0)"
			)

			self.architecture.signal.add(
				name		= "local_out",
				signal_type	= "signed(N downto 0)"
			)

			# Extended inputs
			self.architecture.bodyCodeHeader.add(
				"local_in0 <= in0(N-1) & in0;"
			)
			self.architecture.bodyCodeHeader.add(
				"local_in1 <= in1(N-1) & in1;"
			)

			# Add/sub
			self.architecture.processes.add("sat_add_sub")
			self.architecture.processes["sat_add_sub"].sensitivity_list.add(
					"local_in0")
			self.architecture.processes["sat_add_sub"].sensitivity_list.add(
					"local_in1")
			self.architecture.processes["sat_add_sub"].sensitivity_list.add(
					"local_out")
			self.architecture.processes["sat_add_sub"].sensitivity_list.add(
					"add_or_sub")
			self.architecture.processes["sat_add_sub"].if_list.add()
			self.architecture.processes["sat_add_sub"].if_list[0]._if_.\
					conditions.add("add_or_sub = '0'")
			self.architecture.processes["sat_add_sub"].if_list[0]._if_.body.\
					add("local_out <= local_in0 + local_in1;")
			self.architecture.processes["sat_add_sub"].if_list[0]._else_.\
					body.add("local_out <= local_in0 - local_in1;")

			# Saturate
			self.architecture.processes["sat_add_sub"].if_list.add()
			self.architecture.processes["sat_add_sub"].if_list[1]._if_.\
					conditions.add("local_out(N) /= local_out(N-1)")

			sat_if = If()
			sat_if._if_.conditions.add("local_out(N) = '1'")
			sat_if._if_.body.add("add_sub_out <= to_signed(sat_down, N);")
			sat_if._else_.body.add("add_sub_out <= to_signed(sat_up, N);")

			self.architecture.processes["sat_add_sub"].if_list[1]._if_.body.\
					add(sat_if)
			self.architecture.processes["sat_add_sub"].if_list[1]._else_.body.\
					add("add_sub_out <= local_out(N-1 downto 0);")


		else:

			# Add/sub process
			self.architecture.processes.add("sat_add_sub")
			self.architecture.processes["sat_add_sub"].sensitivity_list.add(
					"in0")
			self.architecture.processes["sat_add_sub"].sensitivity_list.add(
					"in1")
			self.architecture.processes["sat_add_sub"].sensitivity_list.add(
					"add_or_sub")
			self.architecture.processes["sat_add_sub"].if_list.add()
			self.architecture.processes["sat_add_sub"].if_list[0]._if_.\
					conditions.add("add_or_sub = '0'")
			self.architecture.processes["sat_add_sub"].if_list[0]._if_.body.\
					add("add_sub_out <= in0 + in1;")
			self.architecture.processes["sat_add_sub"].if_list[0]._else_.\
					body.add("add_sub_out <= in0 - in1;")


		# Debug
		if debug:
			debug_component(self, debug_list)
