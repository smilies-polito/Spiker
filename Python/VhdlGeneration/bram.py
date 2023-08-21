import subprocess as sp
from math import log2

from headers import bram_header, bram_description
from utils import int_to_hex, floor_pow2, ceil_pow2, n_bytes

import path_config

from vhdl_block import VHDLblock

class Bram(VHDLblock):

	def __init__(self, bram_size = "36Kb", device = "7series", do_reg = 0,
			init_value = 0, init_file = "none", write_width = 72,
			read_width = 72, sim_collision_check = "all", srval = 0,
			write_mode = "read_first"):

		self.bram_tot_bit 	= [18432, 36864]
		self.max_word_width_list 	= [36, 72]

		self.bram_size_list = [str(int(size // 1000)) + "Kb" for size in
				self.bram_tot_bit]

		self.n_parity = 1

		self.device_list = ["7series", "virtex5", "virtex6", 
					"spartan6"]

		self.write_mode_list = ["write_first", "read_first", 
					"no_change"]

		self.sim_collision_check_list = ["all", "warning_only",
				"generate_x_only", "none"]

		self.init_width_bin = 256
		self.init_width_hex = int(self.init_width_bin // 4)

		if bram_size not in self.bram_size_list:
			raise ValueError("Valid values are ", 
				self.bram_size_list)

		if device not in self.device_list:
			raise ValueError("Allowed devices are ",
					self.device_list)

		if do_reg != 0 and do_reg != 1:
			raise ValueError("Allowed Data Output Register "
					"values are 0 and 1")

		if init_value < 0 or init_value > self.max_word_width_list[-1]:
			raise ValueError("Allowed init values for output "
					"register are between 0 and " +
					str(self.max_word_width_list) + " bit")

		if write_width < 0 or write_width > \
		self.max_word_width_list[self.bram_size_list.index(bram_size)]:
			raise ValueError("Allowed Write Width between 1 "
				"and 72. Values from 37 to 72 allowed only "
				"if BRAM size is 36 Kb")

		if read_width < 0 or read_width > \
		self.max_word_width_list[self.bram_size_list.index(bram_size)]:
			raise ValueError("Allowed Read Width between 1 "
				"and 72. Values from 37 to 72 allowed only "
				"if BRAM size is 36 Kb")

		if sim_collision_check not in self.sim_collision_check_list:
			raise ValueError("Valid values are ", 
				self.sim_collision_check_list)


		if srval < 0 or srval > self.max_word_width_list[-1]:
			raise ValueError("Allowed set/reset values for output "
					"register are between 0 and " +
					str(self.max_word_width_list) + " bit")

		if write_mode not in self.write_mode_list:
			raise ValueError("Valid values are ", 
				self.write_mode_list)


		# Target BRAM
		self.bram_size	= bram_size

		# Target device
		self.device	= device

		# Optional output register
		self.do_reg	= do_reg

		# Initial values on the output port
		self.init_val_int	= init_value
		self.init_value	= int_to_hex(
				init_value, 
				width = int(self.max_word_width_list[-1] // 4))

		self.init_file = init_file

		self.write_width = write_width

		self.read_width = read_width

		self.srval_int = srval
		self.srval = int_to_hex(
				srval, 
				width = int(self.max_word_width_list[-1] // 4))

		self.write_mode = write_mode


		width = max(self.read_width, self.write_width)

		self.bram_depth = floor_pow2(self.bram_tot_bit[
			self.bram_size_list.index(
				self.bram_size)]) // \
			ceil_pow2(width - int(width // (8 + self.n_parity)))

		self.addr_width = int(log2(self.bram_depth))

		self.we_width = ceil_pow2(n_bytes(write_width - int(write_width
			// 8)))

		self.sim_collision_check = sim_collision_check


		self.bram_table = self.shape_table()		
		max_addr_width = 0

		for size in self.bram_table:
			for key in self.bram_table[size]:
				if key == "addr_width":
					tmp_max_width = \
					max(self.bram_table[size][key])

					if tmp_max_width > \
					max_addr_width:
						max_addr_width = \
						tmp_max_width
					

		self.max_addr_width = max_addr_width



		VHDLblock.__init__(self, "bram_sdp_macro")

		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		self.postLibraryHeader.add(bram_description)

		self.entity.generic.add(
			name 		= "bram_size",
			gen_type 	= "string",
			value		= "\"" + self.bram_size + "\"")
		self.entity.generic.add(
			name		= "device",
			gen_type 	= "string",
			value		= "\"" + self.device + "\"")
		self.entity.generic.add(
			name		= "do_reg",
			gen_type 	= "integer",
			value		= str(self.do_reg))
		self.entity.generic.add(
			name		= "init",
			gen_type 	= "bit_vector(" +
			str(self.max_word_width_list[-1]-1) + " downto 0)",
			value		= str("X\"" + self.init_value + "\""))
		self.entity.generic.add(
			name		= "init_file",
			gen_type 	= "string",
			value		= str("\"" + self.init_file + "\""))
		self.entity.generic.add(
			name		= "write_width",
			gen_type 	= "integer",
			value		= str(self.write_width))
		self.entity.generic.add(
			name		= "read_width",
			gen_type 	= "integer",
			value		= str(self.read_width))
		self.entity.generic.add(
			name		= "srval",
			gen_type 	= "bit_vector(" +
			str(self.max_word_width_list[-1]-1) + " downto 0)",
			value		= str("X\"" + self.srval + "\""))
		self.entity.generic.add(
			name 		= "write_mode",
			gen_type 	= "string",
			value		= "\"" + self.write_mode + "\"")
		self.entity.generic.add(
			name 		= "sim_collision_check",
			gen_type 	= "string",
			value		= "\"" + self.sim_collision_check + \
					"\"")


		for i in range(floor_pow2(self.bram_tot_bit[-1]) //
				self.init_width_bin):

			self.entity.generic.add(
				name = "init_" + int_to_hex(i, width = 2),
				gen_type 	= "std_logic_vector(" + \
					str(self.init_width_bin-1) + \
					" downto 0)",
				value		= str("X\"" +
					self.init_width_hex*"0" + "\"")
			)
		

		self.entity.port.add(
			name		= "do",
			direction	= "out",
			port_type	= "std_logic_vector(" + 
					str(self.read_width-1) + " downto 0)")
		self.entity.port.add(
			name		= "rdclk",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name		= "rdaddr",
			direction	= "in",
			port_type	= "std_logic_vector(" +
					str(self.max_addr_width-1) + 
					" downto 0)")
		self.entity.port.add(
			name		= "rden",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name		= "regce",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name		= "rst",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name		= "we",
			direction	= "in",
			port_type	= "std_logic_vector(" +
					str(self.we_width-1) + " downto 0)")
		self.entity.port.add(
			name		= "wrclk",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name		= "wraddr",
			direction	= "in",
			port_type	= "std_logic_vector(" +
					str(self.max_addr_width-1) + 
					" downto 0)")
		self.entity.port.add(
			name		= "wren",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name		= "di",
			direction	= "in",
			port_type	= "std_logic_vector(" +
					str(self.write_width-1) + " downto 0)")

	def shape_table(self):

		table = {}

		for size in self.bram_size_list:
			table[size] = {}

			table[size]["min_width"] = []
			table[size]["max_width"] = []
			table[size]["bram_depth"] = []
			table[size]["addr_width"] = []
			table[size]["we_width"] = []

		for size, max_word_width_list in sorted(zip(self.bram_size_list,
		self.max_word_width_list), reverse = True):
			
			count = 0
			size_count = 0

			for word_width in range(max_word_width_list, 0, -1):


				bram_depth = floor_pow2(self.bram_tot_bit[
					self.bram_size_list.index(size)]) \
					// ceil_pow2(word_width - 
					int(word_width // (8 + self.n_parity)))

				addr_width = int(
					log2(bram_depth))

				we_width = ceil_pow2(n_bytes(
					word_width - int(word_width
					// (8 + self.n_parity))))
				
				if not table[size]["bram_depth"] or \
				bram_depth != table[size]["bram_depth"][-1]:

					if table[size]["max_width"]:
						table[size]["min_width"].append(
						table[size][ "max_width"][-1] - 
						count)

					table[size]["max_width"].append(
						word_width)
					table[size]["bram_depth"].append(
						bram_depth)
					table[size]["addr_width"].append(
						addr_width)
					table[size]["we_width"].append(we_width)
					

					count = 0

				else:
					count += 1

			if table[size]["max_width"]:
				table[size]["min_width"].append(table[size][
					"max_width"][-1] - count)

		return table
