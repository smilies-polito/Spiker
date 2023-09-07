import numpy as np
import torch

from math import log2
from typing import Union

from utils import fixed_point_array, ceil_pow2, int_to_hex, int_to_bin
from vhdl import sub_components, debug_component, coe_file

import path_config
from vhdl_block import VHDLblock


class Rom(VHDLblock):

	def __init__(self, init_array : Union[np.ndarray, torch.Tensor],
			bitwidth : int, fp_decimals : int = 0,
			max_word_size : int = 4608, max_depth : int = 1048576,
			init_file : str = None, name_term : str = "", debug =
			False, debug_list = []): 

		self.name_term = name_term
		self.name = "rom_" + str(init_array.shape[1]) + "x" + \
			str(init_array.shape[0]) + self.name_term

		self.rom_columns	= init_array.shape[0]
		self.rom_rows		= init_array.shape[1]
		self.addr_width		= int(log2(ceil_pow2(self.rom_rows)))

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
			self.init_file = self.name + ".coe"
		else:
			self.init_file = init_file

		self.components = sub_components(self)

		super().__init__(self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def initialize(self, output_dir = "output"):

		fp_array = fixed_point_array(
			self.init_array, 
			self.bitwidth,
			self.fp_decimals,
			"signed"
		)

		rows = []

		for j in range(self.rom_rows):

			rom_row = ""

			for i in range(self.rom_columns):

				bin_weight = int_to_bin(fp_array[i][j], width =
						self.bitwidth)

				rom_row += bin_weight

			rows.append(rom_row)

		coe_file(rows, self.init_file, output_dir = output_dir)


	def vhdl(self, debug = False, debug_list = []):

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
					str(self.addr_width-1)  + " downto 0)"
		)

		for i in range(self.rom_columns):

			hex_width = int(log2(ceil_pow2(self.rom_columns)) // 4)

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
			str(self.bitwidth*self.rom_columns-1)
			+ " downto 0)"
		)

		for i in range(self.rom_columns):

			hex_width = int(log2(ceil_pow2(self.rom_columns)) // 4)

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

		# Debug
		if debug:
			debug_component(self, debug_list)


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
					str(self.addr_width - 1) + " downto 0)"
		)

		self.rom_ip.entity.port.add(
			name		= "douta",
			direction	= "out",
			port_type	= "std_logic_vector(" +
			str(self.bitwidth*self.rom_columns-1)
			+ " downto 0)"
		)

	def write_file(self, output_dir = "output"):
		super().write_file(output_dir = output_dir)
		self.initialize(output_dir = output_dir)
