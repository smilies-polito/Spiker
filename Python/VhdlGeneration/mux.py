import subprocess as sp
from math import log2

from utils import ceil_pow2

import path_config

from vhdl_block import VHDLblock

class Mux(VHDLblock):

	def __init__(self, n_in = 8, in_type = "std_logic_vector",
			bitwidth = 4):

		if type(n_in) is not int or n_in < 2:

			print("Invalid number of inputs for multiplexer\n")
			exit(-1)

		elif type(in_type) is not str or in_type != "std_logic" and \
			in_type != "std_logic_vector" and in_type != "signed" \
			and in_type != "unsigned":

			print("Invalid input type for multiplexer\n")
			exit(-1)

		elif type(bitwidth) is not int or bitwidth < 1 or \
			bitwidth == 1 and in_type != "std_logic" or \
			bitwidth > 1 and in_type == "std_logic":

			print("Invalid input bitwidth in multiplexer")
			exit(-1)

		n_in = ceil_pow2(n_in)
		n_sel = int(log2(n_in))

		name = "mux_" + str(n_in) + "to1"

		if in_type != "std_logic" and in_type != "std_logic_vector":
			name = name + "_" + in_type

		VHDLblock.__init__(self, entity_name = name)

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		if in_type != "std_logic" and in_type != "std_logic_vector":
			self.library["ieee"].package.add("numeric_std")

		if bitwidth > 1:
			self.entity.generic.add("bitwidth", "integer",
					str(bitwidth))

		# Input ports
		if n_sel == 1:
			sel_type = "std_logic"
			quote = "\'"
		else:
			sel_type = "std_logic_vector(" + str(n_sel-1) + \
					" downto 0)"
			quote = "\""

		self.entity.port.add("mux_sel", "in", sel_type)

		for port_number in range(n_in):

			port_name = "in" + str(port_number)

			if in_type == "std_logic":
				port_type = in_type
			else:
				port_type = in_type + "(bitwidth-1 downto 0)"

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
		
		for port_number in range(n_in - 1):

			port_name = "in" + str(port_number)

			sel_value = quote + "{0:{fill}{width}{base}}".format(
					port_number, 
					fill = 0, 
					width = n_sel, 
					base = "b") + quote

			self.architecture.processes["selection"].\
				case_list["mux_sel"].when_list.add(sel_value)

			self.architecture.processes["selection"].\
				case_list["mux_sel"].when_list[sel_value].\
				body.add("mux_out <= " + port_name + ";")

		
		port_name = "in" + str(n_in - 1)

		self.architecture.processes["selection"].case_list["mux_sel"].\
			others.body.add("mux_out <= " + port_name + ";")


	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl " + self.entity.name + ".vhd" + "; "

		sp.run(command, shell = True)

		print("\n")
	

	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")
