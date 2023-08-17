from bram import Bram

import path_config
from vhdl_block import VHDLblock


class Memory(VHDLblock):

	def __init__(self, depth, tot_elements, bitwidth, mem_type = "bram"):

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
						["max_width"][depth_index] < \
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
						["max_width"][depth_index] < \
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
