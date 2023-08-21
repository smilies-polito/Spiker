import subprocess as sp
from math import log2

from bram import Bram
from utils import int_to_hex, ceil_pow2

import path_config
from vhdl_block import VHDLblock


class Memory(VHDLblock):

	def __init__(self, depth, tot_elements, bitwidth, mem_type = "bram",
			max_n_bram = 140):

		if mem_type == "bram":
			self.bram = Bram()

			self.bram_table = self.bram.shape_table()

			too_big = True
			too_small = False
			size_index = 0

			final_size = ""
			
			bram_tot_bit = sorted(self.bram.bram_tot_bit, 
					reverse = True)
				
			while too_big and not too_small and size_index < \
			len(bram_tot_bit):

				size = bram_tot_bit[size_index]

				key = str(int(size // 1000)) + "Kb"

				stop = False
				depth_index = 0

				while not stop and depth_index < len(
				self.bram_table[key]["bram_depth"]):

					if self.bram_table[key]\
					["bram_depth"][depth_index] > depth:

						if self.bram_table[key]\
						["max_width"][depth_index] - \
						self.bram_table[key]\
						["max_width"][depth_index] // \
						(8 + self.bram.n_parity) < \
						bitwidth:
							too_small = True


						else:
							stop = True

							final_size = key
							final_word_width = \
							self.bram_table[key]\
							["max_width"]\
							[depth_index]
							final_addr_width = \
							self.bram_table[key]\
							["addr_width"]\
							[depth_index]
							final_we_width = \
							self.bram_table[key]\
							["we_width"]\
							[depth_index]
							final_depth = \
							self.bram_table[key]\
							["bram_depth"][\
							depth_index]

						if self.bram_table[key]\
						["max_width"][depth_index] - \
						self.bram_table[key]\
						["max_width"][depth_index] // \
						(8 + self.bram.n_parity) < \
						bitwidth*tot_elements:

							too_big = False

					depth_index += 1

				size_index += 1

			if not final_size:
				raise ValueError("Cannot fit the BRAM. "
					"Data are too big")


			self.size = final_size
			self.word_width = final_word_width
			self.addr_width = final_addr_width
			self.we_width = final_we_width
			self.depth = final_depth

			self.max_word_width = max(self.bram.max_word_width_list)


			self.el_per_word = int(self.word_width -\
				self.word_width //
				(8 + self.bram.n_parity) // 
				bitwidth) 

			self.spare_elements = tot_elements % \
				self.el_per_word


			self.n_bram = int(tot_elements //
				self.el_per_word)

			if self.spare_elements:
				self.n_bram += 1

			if self.n_bram > max_n_bram:
				raise ValueError("Cannot fit the maximum "
						"amount of BRAMs. Needed: %d. "
						"Available: %d" %(self.n_bram,
							max_n_bram))


		VHDLblock.__init__(self, "memory")

		self.bram = Bram(
			bram_size = self.size,
			device = self.bram.device,
			do_reg = self.bram.do_reg,
			init_value = self.bram.init_val_int,
			init_file = self.bram.init_file,
			write_width = self.word_width,
			read_width = self.word_width,
			sim_collision_check = self.bram.sim_collision_check,
			srval = self.bram.srval_int,
			write_mode = self.bram.write_mode	
		)
						

		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library.add("unisim")
		self.library["unisim"].package.add("vcomponents")
		self.library.add("unimacro")
		self.library["unimacro"].package.add("vcomponents")


		for key in self.bram.entity.port:
			if self.bram.entity.port[key].direction == "in":
				self.entity.port.add(
					name = self.bram.entity.port[key].name,
					direction = self.bram.entity.port[key].\
						direction,
					port_type = self.bram.entity.port[key].\
						port_type
				)

		for out_index in range(tot_elements):
			self.entity.port.add(
				name = "do_" + int_to_hex(out_index, width =
					int(log2(ceil_pow2(tot_elements)))
					// 4),
				direction = "out",
				port_type = "std_logic_vector(" +
				str(bitwidth-1) + " downto 0)"
			)


		self.architecture.constant.add(
			name 		= "bram_size",
			const_type 	= "string",
			value		= "\"" + self.size + "\"")
		self.architecture.constant.add(
			name		= "device",
			const_type 	= "string",
			value		= "\"" + self.bram.device + "\"")
		self.architecture.constant.add(
			name		= "do_reg",
			const_type 	= "integer",
			value		= str(self.bram.do_reg))
		self.architecture.constant.add(
			name		= "init",
			const_type 	= "bit_vector(" +
			str(self.max_word_width-1) + " downto 0)",
			value		= str("X\"" + self.bram.init_value + 
						"\""))
		self.architecture.constant.add(
			name		= "init_file",
			const_type 	= "string",
			value		= str("\"" + self.bram.init_file + 
						"\""))
		self.architecture.constant.add(
			name		= "srval",
			const_type 	= "bit_vector(" +
			str(self.max_word_width-1) + " downto 0)",
			value		= str("X\"" + self.bram.srval + "\""))
		self.architecture.constant.add(
			name 		= "write_mode",
			const_type 	= "string",
			value		= "\"" + self.bram.write_mode + "\"")
		self.architecture.constant.add(
			name 		= "sim_collision_check",
			const_type 	= "string",
			value		= "\"" + self.bram.sim_collision_check \
						+ "\"")

		self.architecture.constant.add(
			name 		= "bit_width",
			const_type	= "integer",
			value		= str(bitwidth)
		)
		self.architecture.constant.add(
			name 		= "n_bram",
			const_type	= "integer",
			value		= str(self.n_bram)
		)
		self.architecture.constant.add(
			name 		= "elements_per_word",
			const_type	= "integer",
			value		= str(self.el_per_word)
		)
		self.architecture.constant.add(
			name		= "write_width",
			const_type	= "integer",
			value 		= str(self.word_width)
		)
		self.architecture.constant.add(
			name		= "read_width",
			const_type	= "integer",
			value 		= str(self.word_width)
		)
		self.architecture.constant.add(
			name		= "addr_width",
			const_type	= "integer",
			value 		= str(self.bram.max_addr_width)
		)
		self.architecture.constant.add(
			name		= "we_width",
			const_type	= "integer",
			value 		= str(self.we_width)
		)



		for i in range(self.n_bram):

			self.architecture.signal.add(
				name		= "do_bram_" + str(i),
				signal_type	= "std_logic_vector("
						"read_width-1 downto 0)"
			)

			self.architecture.instances.add(
				self.bram, "bram_" + str(i))


			self.architecture.instances["bram_" + 
				str(i)].generic_map(mode = "no")

			for g_map in self.bram.entity.generic:
				if "init" not in g_map:
					self.architecture.instances["bram_" +
						str(i)].g_map.add(g_map, g_map)

			self.architecture.instances["bram_" + 
				str(i)].port_map()
			self.architecture.instances["bram_" + 
				str(i)].p_map.add("do", "do_bram_" + str(i))




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






a = Memory(100, 1, 32)

print(a.code())
a.write_file()
a.compile()
a.elaborate()
