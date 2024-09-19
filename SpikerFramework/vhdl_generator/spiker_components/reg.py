from .vhdl import sub_components, debug_component

from .vhdltools.if_statement import If
from .vhdltools.vhdl_block import VHDLblock

class Reg(VHDLblock):

	def __init__(self, bitwidth = 8, reg_type = "std_logic", rst =
			None, active = "low", debug = False, debug_list = []):

		self.name = "reg"

		if reg_type != "std_logic_vector" and reg_type != "std_logic":
			self.name = self.name + "_" + reg_type

		if rst:
			self.name = self.name + "_" + rst + "_rst"

		if reg_type != "std_logic" and reg_type != "std_logic_vector" \
		and reg_type != "signed" and  reg_type != "unsigned":
			raise ValueError("Invalid signal type in " + self.name)

		self.bitwidth = bitwidth
		self.reg_type = reg_type
		self.rst = rst
		self.active = active
		self.components = sub_components(self)

		VHDLblock.__init__(self, entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		if self.reg_type != "std_logic" and \
		self.reg_type !=  "std_logic_vector":
			self.library["ieee"].package.add("numeric_std")

		# Generics
		if self.reg_type != "std_logic":
			self.entity.generic.add("N", "integer", str(self.bitwidth))

		# Input ports
		self.entity.port.add("clk", "in", "std_logic")
		self.entity.port.add("en", "in", "std_logic")
		
		if self.rst:
			self.rst_name = "rst"

			if self.active == "low":
				self.rst_name = self.rst_name + "_n"

			self.entity.port.add(self.rst_name, "in", "std_logic")

		if self.reg_type == "std_logic":
			self.entity.port.add("reg_in", "in", 
				"std_logic")
			
			# Output ports
			self.entity.port.add("reg_out", "out", 
				"std_logic")

		else:
			self.entity.port.add("reg_in", "in", 
				self.reg_type + "(N-1 downto 0)")
			
			# Output ports
			self.entity.port.add("reg_out", "out", 
				self.reg_type + "(N-1 downto 0)")

		self.architecture.processes.add("sample")
		self.architecture.processes["sample"].sensitivity_list.add(
				"clk")
		self.architecture.processes["sample"].sensitivity_list.add("en")

		self.architecture.processes["sample"].if_list.add()
		self.architecture.processes["sample"].if_list[0]._if_.\
				conditions.add("clk'event")
		self.architecture.processes["sample"].if_list[0]._if_.\
				conditions.add("clk = '1'", "and")

		# Sample process
		if self.rst:

			self.architecture.processes["sample"].sensitivity_list.\
					add(self.rst_name)


			if self.rst == "sync":

				# Reset inner if statement
				reset_if = If()

				if self.active == "low":
					self.rst_value = "\'0\'"
				else:
					self.rst_value = "\'1\'"

				reset_if._if_.conditions.add(self.rst_name + " = " +
						self.rst_value)

				reset_if._if_.body.add("reg_out <= "
					"(others => '0');")
				reset_if._elsif_.add()
				reset_if._elsif_[0].conditions.add("en = '1'")
				reset_if._elsif_[0].body.add(
					"reg_out <= reg_in;")

				self.architecture.processes["sample"].\
						if_list[0]._if_.body.add(
						reset_if)

		else:
			self.architecture.processes["sample"].\
					if_list[0]._if_.\
					conditions.add("en = '1'", 
					"and")

			self.architecture.processes["sample"].\
					if_list[0]._if_.body.add(
					"reg_out <= reg_in;")

		# Debug
		if debug:
			debug_component(self, debug_list)
