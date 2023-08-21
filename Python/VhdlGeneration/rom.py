import subprocess as sp
import numpy as np
import torch

from math import log
from typing import Union

from utils import fixed_point_array, ceil_pow2, int_to_hex

import path_config
from vhdl_block import VHDLblock


class Rom(VHDLblock):

	def __init__(self, init_array : Union[np.ndarray, torch.Tensor],
			bitwidth : int, fp_decimals : int = None,
			max_word_size : int = 4608, max_depth : int = 1048576,
			init_file : str = None, name_term : str = ""): 

		self.name_term = name_term

		VHDLblock.__init__(self, "rom" + name_term)

		self.rom_columns	= init_array.shape[0]
		self.rom_rows		= init_array.shape[1]

		if self.rom_columns*bitwidth > max_word_size:
			raise ValueError("Cannot fit ROM. Data are too large")

		if self.rom_rows > max_depth:
			raise ValueError("Cannot fit ROM. Data are too deep")

		self.init_array = init_array
		self.bitwidth	= bitwidth

		if fp_decimals == None:
			self.fp_decimals = bitwidth - 1
		else:
			self.fp_decimals = fp_decimals

		if not init_file:
			self.init_file = self.entity.name + ".coe"
		else:
			self.init_file = init_file

		self.vhdl()
		self.initialize()

	def initialize(self):

		fp_array = fixed_point_array(
			self.init_array, 
			self.bitwidth,
			self.fp_decimals,
			"signed"
		)

		rom_rows = []

		for j in range(fp_array.shape[1]):

			rom_row = ""

			for i in range(fp_array.shape[0]):

				if fp_array[i][j] < 0:
					fill = 1
				else:
					fill = 0

				bin_weight = "{0:{fill}{width}{base}}".\
				format(fp_array[i][j], fill = fill, 
				width = bitwidth, base = "b")

				rom_row += bin_weight

			rom_row += "\n"
			rom_rows.append(rom_row)



		with open(self.init_file, "w") as fp:
			for row in rom_rows:
				fp.write(row)

		return rom_rows

	def vhdl(self):

		self.ip()

		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		self.entity.port.add(
			name 		= "clka",
			direction	= "in",
			port_type	= "std_logic"
		)
		self.entity.port.add(
			name 		= "addra",
			direction	= "in",
			port_type	= "std_logic_vector(" +
			str(int(log(ceil_pow2(self.init_array.shape[0]), 2)))  + 
			" downto 0)"
		)

		for i in range(self.init_array.shape[1]):

			hex_width = int(log(ceil_pow2(self.init_array.shape[0]),
					16))

			if hex_width == 0:
				hex_width = 1

			hex_index = str(int_to_hex(i, width = hex_width))

			self.entity.port.add(
				name 		= "dout_" + hex_index,
				direction	= "out",
				port_type	= "std_logic_vector(" +
						str(self.bitwidth-1) + 
						" downto 0)"
			)

		self.architecture.signal.add(
			name	= "douta",
			signal_type	= "std_logic_vector(" +
			str(self.bitwidth*self.init_array.shape[1]-1)
			+ " downto 0)"
		)

		for i in range(self.init_array.shape[1]):

			hex_width = int(log(ceil_pow2(self.init_array.shape[0]),
					16))

			if hex_width == 0:
				hex_width = 1

			hex_index = str(int_to_hex(i, width = hex_width))

			self.architecture.bodyCodeHeader.add(
				"dout_" + hex_index + " <= douta("
				+ str(self.bitwidth*(i+1)-1) + " downto " + 
				str(self.bitwidth*i) + ");")


		self.architecture.component.add(self.rom_ip)
		self.architecture.instances.add(self.rom_ip, 
			self.entity.name + "_ip_instance")
		self.architecture.instances[self.entity.name + 
			"_ip_instance"].port_map()


	def ip(self):
		self.rom_ip = VHDLblock(self.entity.name + "_ip")

		self.rom_ip.entity.port.add(
			name 		= "clka",
			direction	= "in",
			port_type	= "std_logic"
		)
		self.rom_ip.entity.port.add(
			name 		= "addra",
			direction	= "in",
			port_type	= "std_logic_vector(" +
			str(int(log(ceil_pow2(self.init_array.shape[0]), 2))) + 
			" downto 0)"
		)

		self.rom_ip.entity.port.add(
			name		= "douta",
			direction	= "out",
			port_type	= "std_logic_vector(" +
			str(self.bitwidth*self.init_array.shape[1]-1)
			+ " downto 0)"
		)




	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")


	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")
