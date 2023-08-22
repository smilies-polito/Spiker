import path_config
from if_statement import If
from vhdl_block import VHDLblock

class Reg(VHDLblock):

	def __init__(self, bitwidth = 8, reg_type = "std_logic", rst =
			None, active = "low"):

		if reg_type != "std_logic" and reg_type != "signed" and \
			reg_type != "unsigned":

			print("Invalid register type")
			exit(-1)

		name = "reg"

		if reg_type != "std_logic":
			name = name + "_" + reg_type

		if rst:
			name = name + "_" + rst + "_rst"

		VHDLblock.__init__(self, entity_name = name)

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		if reg_type != "std_logic":
			self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add("N", "integer", str(bitwidth))

		# Input ports
		self.entity.port.add("clk", "in", "std_logic")
		self.entity.port.add("en", "in", "std_logic")
		
		if rst:
			rst_name = "rst"

			if active == "low":
				rst_name = rst_name + "_n"

			self.entity.port.add(rst_name, "in", "std_logic")

		if reg_type == "std_logic":
			self.entity.port.add("reg_in", "in", 
				"std_logic_vector(N-1 downto 0)")
			
			# Output ports
			self.entity.port.add("reg_out", "out", 
				"std_logic_vector(N-1 downto 0)")

		else:
			self.entity.port.add("reg_in", "in", 
				reg_type + "(N-1 downto 0)")
			
			# Output ports
			self.entity.port.add("reg_out", "out", 
				reg_type + "(N-1 downto 0)")

		self.architecture.processes.add("sample")
		self.architecture.processes["sample"].sensitivity_list.add(
				"clk")
		self.architecture.processes["sample"].sensitivity_list.add("en")

		self.architecture.processes["sample"].if_list.add()
		self.architecture.processes["sample"].if_list[0]._if_.\
				conditions.add("clk'event")
		self.architecture.processes["sample"].if_list[0]._if_.\
				conditions.add("clk = '1'", "and")

		# Add/sub process
		if rst:

			self.architecture.processes["sample"].sensitivity_list.\
					add(rst_name)


			if rst == "sync":

				# Reset inner if statement
				reset_if = If()

				if active == "low":
					rst_value = "\'0\'"
				else:
					rst_value = "\'1\'"

				reset_if._if_.conditions.add(rst_name + " = " +
						rst_value)

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
