import subprocess as sp
from math import log2

from headers import bram_header, bram_description
from utils import int_to_hex, floor_pow2, ceil_pow2, n_bytes

import path_config

from vhdl_block import VHDLblock

class Bram(VHDLblock):

	def __init__(self, bram_size = "36Kb", device = "7series", do_reg = 0,
			init_value = 0, init_file = "none", write_width = 72,
			read_width = 72, srval = 0, write_mode = "write_first"):

		self.bram_kb = [18, 36]
		self.bram_size_list = [str(size) + "Kb" for size in
				self.bram_kb]

		self.device_list = ["7series", "virtex5", "virtex6", 
					"spartan6"]
		self.max_word_width = 72
		self.write_mode_list = ["write_first", "read_first", 
					"no_change"]

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

		if init_value < 0 or init_value > self.max_word_width:
			raise ValueError("Allowed init values for output "
					"register are between 0 and " +
					str(self.max_word_width) + " bit")

		if write_width < 0 or bram_size == "18Kb" and write_width > \
		int(self.max_word_width // 2) or bram_size == "36Kb" and \
		write_width > int(self.max_word_width):
			raise ValueError("Allowed Write Width between 1 "
				"and 72. Values from 37 to 72 allowed only "
				"if BRAM size is 36 Kb")

		if read_width < 0 or bram_size == "18Kb" and read_width > \
		int(self.max_word_width // 2) or bram_size == "36Kb" and \
		read_width > int(self.max_word_width):
			raise ValueError("Allowed Read Width between 1 "
				"and 72. Values from 37 to 72 allowed only "
				"if BRAM size is 36 Kb")

		if srval < 0 or srval > self.max_word_width:
			raise ValueError("Allowed set/reset values for output "
					"register are between 0 and " +
					str(self.max_word_width) + " bit")

		if write_mode not in self.write_mode_list:
			raise ValueError("Valid values are ", 
				self.write_mode_list)


		self.bram_depth = bram_size

		# Target BRAM
		self.bram_size	= bram_size

		# Target device
		self.device	= device

		# Optional output register
		self.do_reg	= do_reg

		# Initial values on the output port
		self.init_value	= int_to_hex(
				init_value, 
				width = int(self.max_word_width // 4))

		self.init_file = init_file

		self.write_width = write_width

		self.read_width = read_width

		self.srval = int_to_hex(
				srval, 
				width = int(self.max_word_width // 4))

		self.write_mode = write_mode


		width = max(self.read_width, self.write_width)

		self.bram_depth = floor_pow2(self.bram_kb[
			self.bram_size_list.index(
				self.bram_size)]*1000) // \
			ceil_pow2(width - int(width // 8))

		self.addr_width = int(log2(self.bram_depth))

		self.we_width = ceil_pow2(n_bytes(write_width - int(write_width
			// 8)))

		VHDLblock.__init__(self, "bram_single_macro")

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
			gen_type 	= "std_logic_vector(" +
			str(self.max_word_width-1) + " downto 0)",
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
			gen_type 	= "std_logic_vector(" +
			str(self.max_word_width-1) + " downto 0)",
			value		= str("X\"" + self.srval + "\""))
		self.entity.generic.add(
			name 		= "write_mode",
			gen_type 	= "string",
			value		= "\"" + self.write_mode + "\"")


		for i in range(floor_pow2(self.bram_kb[-1]*1000) //
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
			name		= "clk",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name		= "addr",
			direction	= "in",
			port_type	= "std_logic_vector(" +
					str(self.addr_width-1) + " downto 0)")
		self.entity.port.add(
			name		= "en",
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
			name		= "di",
			direction	= "in",
			port_type	= "std_logic_vector(" +
					str(write_width-1) + " downto 0)")

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

	
def comp(component, output_dir = "output"):

	print("\nCompiling component %s\n"
			%(component.entity.name))

	command = "cd " + output_dir + "; "
	command = command + "xvhdl --2008 " + component.entity.name + ".vhd"

	sp.run(command, shell = True)

	print("\n")

def elaborate(component, output_dir = "output"):

	print("\nElaborating component %s\n"
			%(component.entity.name))

	command = "cd " + output_dir + "; "
	command = command + "xelab " + component.entity.name

	sp.run(command, shell = True)

	print("\n")
	


from testbench import Testbench

class prova(VHDLblock):


	def __init__(self):
		VHDLblock.__init__(self, "prova")

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

a = prova()
a.library.add("unisim")
a.library["unisim"].package.add("vcomponents")
a.library.add("unimacro")
a.library["unimacro"].package.add("vcomponents")
a.library.add("ieee")
a.library["ieee"].package.add("std_logic_1164")

b = Bram(
	bram_size	= "36Kb",
	device		= "7series",
	do_reg		= 0,
	init_value	= 0,
	write_width	= 72,
	read_width	= 72,
	srval 		= 1

)

for key in b.entity.port:
	a.entity.port.add(
		name = b.entity.port[key].name,
		direction = b.entity.port[key].direction,
		port_type = b.entity.port[key].port_type
	)

a.architecture.constant.add(
	name 		= "bram_size",
	const_type	= "string",
	value		= "\"36Kb\"")
a.architecture.constant.add(
	name 		= "device",
	const_type	= "string",
	value		= "\"7series\"")
a.architecture.constant.add(
	name 		= "do_reg",
	const_type	= "integer",
	value		= "0")
a.architecture.constant.add(
	name 		= "init",
	const_type	= "bit_vector(71 downto 0)",
	value		= "X\"" + 18*"0" + "\"")
a.architecture.constant.add(
	name 		= "init_file",
	const_type	= "string",
	value		= "\"none\"")
a.architecture.constant.add(
	name 		= "write_width",
	const_type	= "integer",
	value		= "72")
a.architecture.constant.add(
	name 		= "read_width",
	const_type	= "integer",
	value		= "72")
a.architecture.constant.add(
	name 		= "write_mode",
	const_type	= "string",
	value		= "\"write_first\"")
a.architecture.constant.add(
	name 		= "srval",
	const_type	= "bit_vector(71 downto 0)",
	value		= "X\"" + 18*"0" + "\"")

a.architecture.instances.add(b, "bram")
a.architecture.instances["bram"].port_map()
a.architecture.instances["bram"].generic_map()
a.architecture.instances["bram"].g_map.add("init_00",
	"X\"00000000000000000000000000000000000000000000000000000000000000ff\"")

for i in range(1, 128):
	a.architecture.instances["bram"].g_map.add("init_" + int_to_hex(i, width
		= 2), "X\"" + 64*"0"  + "\"")


tb = Testbench(a)


tb.architecture.processes["en_gen"].for_list.add("address", 0, 511)
tb.architecture.processes["en_gen"].for_list[0].body.add("en <= \'1\';")
tb.architecture.processes["en_gen"].for_list[0].body.add("addr <= "
		"std_logic_vector(to_unsigned(i, 9));")
tb.architecture.processes["en_gen"].for_list[0].body.add("wait for 20 ns;")
tb.architecture.processes["en_gen"].for_list[0].body.add("en <= \'0\';")
tb.architecture.processes["en_gen"].for_list[0].body.add(
		"addr <= (others => '0');")
tb.architecture.processes["en_gen"].for_list[0].body.add("wait for 20 ns;")

print(tb.code())


tb.write_file_all()
tb.compile_all()
tb.elaborate()
