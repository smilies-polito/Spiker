import numpy as np
import torch

from math import log2
from typing import Union

from .utils import fixed_point_array, ceil_pow2, int_to_hex, int_to_bin
from .vhdl import sub_components, debug_component, coe_file

from .vhdltools.vhdl_block import VHDLblock


class Rom(VHDLblock):

	def __init__(self, init_array : Union[np.ndarray, torch.Tensor],
			bitwidth : int, fp_decimals : int = 0,
			max_word_size : int = np.inf, max_depth : int = np.inf,
			init_file : str = None, name_term : str = "",
			functional = False, writable : bool = False,
			debug = False, debug_list = []):

		# When writable=True the block emits a dual-port RAM instead of a
		# read-only ROM: the entity is renamed ram_<cols>x<rows><label>,
		# the read port is exposed as (clk, raddr), and an extra write port
		# (wea, waddr, din) is added. The initialization path (.coe file +
		# behavioural IP) is reused — the .coe doubles as RAM init data.
		self.writable = writable
		prefix = "ram_" if writable else "rom_"

		self.name_term = name_term
		self.name = prefix + str(init_array.shape[1]) + "x" + \
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
		
		self.functional = functional

		super().__init__(self.name)

		self.initialize()

		if functional:
			self.ip()

		self.components = sub_components(self)

		self.vhdl(debug = debug, debug_list = debug_list)

	def initialize(self):

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

				rom_row = bin_weight + rom_row

			rows.append(rom_row)

		self.rows = rows

	def write_coe(self, output_dir = "output"):

		coe_file(self.rows, self.init_file, output_dir = output_dir)


	def vhdl(self, debug = False, debug_list = []):

		if not self.functional:
			self.ip()

		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		# In writable (RAM) mode, the read port carries different names to
		# avoid colliding with the (separate) write port — match the
		# Spiker-LL fork: clk, raddr, dout_*, wea, waddr, din.
		read_clk_name  = "clk"   if self.writable else "clka"
		read_addr_name = "raddr" if self.writable else "addra"

		self.entity.port.add(
			name 		= read_clk_name,
			direction	= "in",
			port_type	= "std_logic"
		)
		self.entity.port.add(
			name 		= read_addr_name,
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

		if self.writable:
			# Extra write-side ports — driven by the trainer + multi_input
			# at runtime; values come from the on-chip learning module.
			self.entity.port.add(
				name		= "wea",
				direction	= "in",
				port_type	= "std_logic"
			)
			self.entity.port.add(
				name		= "waddr",
				direction	= "in",
				port_type	= "std_logic_vector(" +
						str(self.addr_width-1) + " downto 0)"
			)
			self.entity.port.add(
				name		= "din",
				direction	= "in",
				port_type	= "std_logic_vector(" +
					str(self.bitwidth*self.rom_columns-1) +
					" downto 0)"
			)

		# The Spiker-LL fork names the wrapper's internal read bus after
		# the RAM's true dual-port read output (doutb); the read-only ROM
		# keeps the original single-port name (douta).
		read_bus = "doutb" if self.writable else "douta"

		self.architecture.signal.add(
			name	= read_bus,
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
				"dout_" + hex_index + " <= " + read_bus + "("
				+ str(self.bitwidth*(i+1)-1) + " downto " +
				str(self.bitwidth*i) + ");")


		self.architecture.component.add(self.rom_ip)
		ip_inst_name = self.entity.name + "_ip_instance"
		self.architecture.instances.add(self.rom_ip, ip_inst_name)

		if self.writable:
			# Writable IP has separate write/read ports — wire them
			# explicitly because the names don't match the outer entity.
			self.architecture.instances[ip_inst_name].port_map(mode="no")
			self.architecture.instances[ip_inst_name].p_map.add(
				"clka",  "clk")
			self.architecture.instances[ip_inst_name].p_map.add(
				"wea",   "wea")
			self.architecture.instances[ip_inst_name].p_map.add(
				"addra", "waddr")
			self.architecture.instances[ip_inst_name].p_map.add(
				"dina",  "din")
			self.architecture.instances[ip_inst_name].p_map.add(
				"clkb",  "clk")
			self.architecture.instances[ip_inst_name].p_map.add(
				"addrb", "raddr")
			self.architecture.instances[ip_inst_name].p_map.add(
				"doutb", "doutb")
		else:
			self.architecture.instances[ip_inst_name].port_map()

		# Debug
		if debug:
			debug_component(self, debug_list)


	def ip(self):

		init_matrix = "(\n"

		for i in range(len(self.rows)):
			init_matrix = init_matrix + "\"" + self.rows[i] + \
					"\",\n"

		init_matrix = init_matrix + "\"" + \
			"0"*self.rom_columns*self.bitwidth + "\")"

		self.rom_ip = VHDLblock(self.entity.name + "_ip")

		self.rom_ip.library.add("ieee")
		self.rom_ip.library["ieee"].package.add("std_logic_1164")
		self.rom_ip.library["ieee"].package.add("numeric_std")

		# Storage array — shared between ROM (read-only) and RAM (R/W).
		# The ``mem`` constant doubles as the RAM init value: at synthesis
		# Vivado picks it up exactly the same way it would a .coe file.
		mem_type_name = "ram_type" if self.writable else "rom_type"
		mem_decl = "signal" if self.writable else "constant"

		if self.writable:
			# Dual-port RAM: separate write port (clka/wea/addra/dina) and
			# read port (clkb/addrb/doutb). Behavioural model only;
			# replaced by a Vivado Block Memory IP at synthesis.
			self.rom_ip.entity.port.add(
				name="clka", direction="in", port_type="std_logic")
			self.rom_ip.entity.port.add(
				name="wea",  direction="in", port_type="std_logic")
			self.rom_ip.entity.port.add(
				name="addra", direction="in",
				port_type="std_logic_vector(" +
				str(self.addr_width - 1) + " downto 0)")
			self.rom_ip.entity.port.add(
				name="dina", direction="in",
				port_type="std_logic_vector(" +
				str(self.bitwidth*self.rom_columns-1) + " downto 0)")
			self.rom_ip.entity.port.add(
				name="clkb", direction="in", port_type="std_logic")
			self.rom_ip.entity.port.add(
				name="addrb", direction="in",
				port_type="std_logic_vector(" +
				str(self.addr_width - 1) + " downto 0)")
			self.rom_ip.entity.port.add(
				name="doutb", direction="out",
				port_type="std_logic_vector(" +
				str(self.bitwidth*self.rom_columns-1) + " downto 0)")
		else:
			# Single-port ROM (read-only) — original Spiker behaviour.
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

		self.rom_ip.architecture.customTypes.add(
			mem_type_name,
			"Array",
			"0 to " + str(self.rom_rows),
			"std_logic_vector(" +
			str(self.rom_columns*self.bitwidth-1)
			+ " downto 0)"
		)

		if self.writable:
			# ``mem`` is a signal initialised to the trained weights —
			# write_proc updates entries on the rising edge of clka.
			self.rom_ip.architecture.signal.add(
				name="ram",
				signal_type=mem_type_name + " := " + init_matrix)

			# Write port — synchronous on clka.
			self.rom_ip.architecture.processes.add("write_proc")
			self.rom_ip.architecture.processes["write_proc"].\
				sensitivity_list.add("clka")
			self.rom_ip.architecture.processes["write_proc"].if_list.add()
			self.rom_ip.architecture.processes["write_proc"].\
				if_list[0]._if_.conditions.add("clka'event")
			self.rom_ip.architecture.processes["write_proc"].\
				if_list[0]._if_.conditions.add("clka='1'", "and")
			# Nested if for wea — keep the structure simple by emitting
			# the inner write as a raw body line.
			self.rom_ip.architecture.processes["write_proc"].\
				if_list[0]._if_.body.add(
				"if wea = '1' then\n"
				"            ram(to_integer(unsigned(addra))) <= dina;\n"
				"        end if;")

			# Read port — synchronous on clkb.
			self.rom_ip.architecture.processes.add("read_proc")
			self.rom_ip.architecture.processes["read_proc"].\
				sensitivity_list.add("clkb")
			self.rom_ip.architecture.processes["read_proc"].if_list.add()
			self.rom_ip.architecture.processes["read_proc"].\
				if_list[0]._if_.conditions.add("clkb'event")
			self.rom_ip.architecture.processes["read_proc"].\
				if_list[0]._if_.conditions.add("clkb='1'", "and")
			self.rom_ip.architecture.processes["read_proc"].\
				if_list[0]._if_.body.add(
				"doutb <= ram(to_integer(unsigned(addrb)));")
		else:
			# Read-only ROM — original behavior.
			self.rom_ip.architecture.constant.add(
				"mem", mem_type_name, init_matrix)

			self.rom_ip.architecture.processes.add("rom_behavior")
			self.rom_ip.architecture.processes["rom_behavior"].\
				sensitivity_list.add("clka")
			self.rom_ip.architecture.processes["rom_behavior"].\
				if_list.add()
			self.rom_ip.architecture.processes["rom_behavior"].\
				if_list[0]._if_.conditions.add("clka'event")
			self.rom_ip.architecture.processes["rom_behavior"].\
				if_list[0]._if_.conditions.add("clka='1'", "and")
			self.rom_ip.architecture.processes["rom_behavior"].\
				if_list[0]._if_.body.add(
				"douta <= mem(to_integer(unsigned(addra)));")


	def write_file(self, output_dir = "output", rm = False):
		super().write_file(output_dir = output_dir, rm = rm)

		if self.functional:
			self.rom_ip.write_file(output_dir = output_dir, rm = rm)

		# Writable RAMs always need the .coe as the eventual Vivado BRAM
		# IP's initial contents, regardless of functional/interface mode.
		# Read-only ROMs keep the exact original behaviour (.coe only when
		# NOT also emitting the functional behavioural model), so legacy
		# (non-learning) output is byte-for-byte unchanged.
		if self.writable or not self.functional:
			self.write_coe(output_dir = output_dir)
