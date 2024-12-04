from math import log2

from .cnt import Cnt
from .cmp import Cmp

from .testbench import Testbench
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2, random_binary

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If

class MultiCycleDP(VHDLblock):

	def __init__(self, n_cycles = 10, debug = False, debug_list = []):

		self.name = "multi_cycle_dapapath"

		self.n_cycles = n_cycles
		self.cycles_cnt_bitwidth = int(log2(ceil_pow2(n_cycles + 1))) \
				+ 1

		self.counter = Cnt(bitwidth = self.cycles_cnt_bitwidth) 
		self.cmp = Cmp(bitwidth = self.cycles_cnt_bitwidth, cmp_type =
				"eq", signal_type = "std_logic")

		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add(
			name		= "cycles_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.cycles_cnt_bitwidth))
		self.entity.generic.add(
			name		= "n_cycles", 
			gen_type	= "integer",
			value		= str(self.n_cycles))


		self.entity.port.add(
			name 		= "clk", 
			direction	= "in", 
			port_type	= "std_logic")

		# Input controls
		self.entity.port.add(
			name 		= "cycles_cnt_en", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "cycles_cnt_rst_n", 
			direction	= "in", 
			port_type	= "std_logic")

		# Output
		self.entity.port.add(
			name 		= "stop", 
			direction	= "out", 
			port_type	= "std_logic")

		# Signals
		self.architecture.signal.add(
			name		= "cycles_cnt",
			signal_type	= "std_logic_vector("
					"cycles_cnt_bitwidth-1 downto 0)")

		# Components
		self.architecture.component.add(self.counter)
		self.architecture.component.add(self.cmp)

		# Cycles cnt
		self.architecture.instances.add(self.counter, "cycles_counter")
		self.architecture.instances["cycles_counter"].generic_map()
		self.architecture.instances["cycles_counter"].g_map.add("N",
				"cycles_cnt_bitwidth")
		del(self.architecture.instances["cycles_counter"].\
			g_map["rst_value"])

		self.architecture.instances["cycles_counter"].port_map()
		self.architecture.instances["cycles_counter"].p_map.add(
				"cnt_en", "cycles_cnt_en")
		self.architecture.instances["cycles_counter"].p_map.add(
				"cnt_rst_n", "cycles_cnt_rst_n")
		self.architecture.instances["cycles_counter"].p_map.add(
				"cnt_out", "cycles_cnt")

		# Cycles cmp
		self.architecture.instances.add(self.cmp, "cycles_cmp")
		self.architecture.instances["cycles_cmp"].generic_map()
		self.architecture.instances["cycles_cmp"].g_map.add("N",
				"cycles_cnt_bitwidth")
		self.architecture.instances["cycles_cmp"].port_map()
		self.architecture.instances["cycles_cmp"].p_map.add(
				"in0", "cycles_cnt")
		self.architecture.instances["cycles_cmp"].p_map.add(
				"in1", "std_logic_vector(to_unsigned("
				"n_cycles + 1, cycles_cnt_bitwidth))")
		self.architecture.instances["cycles_cmp"].p_map.add("cmp_out",
				"stop")

		# Debug
		if debug:
			debug_component(self, debug_list)

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)


class MultiCycleDP_tb(Testbench):

	def __init__(self, clock_period = 20, file_output = False, output_dir =
			"output", file_input = False, input_dir = "",
			input_signal_list = [], n_cycles = 10, debug = False, 
			debug_list = []):

		self.n_cycles = n_cycles

		self.dut = MultiCycleDP(
			n_cycles = n_cycles,
			debug = debug,
			debug_list = debug_list
		)

		self.components = sub_components(self)

		super().__init__(
			dut = self.dut, 
			clock_period = clock_period,
			file_output = file_output,
			output_dir = output_dir,
			file_input = file_input,
			input_dir = input_dir,
			input_signal_list = input_signal_list
		)
		
		self.vhdl(clock_period = clock_period, file_output = file_output)

	def vhdl(self, clock_period = 20, file_output = False):

		# Cycle counter enable
		self.architecture.processes["cycles_cnt_en_gen"].bodyHeader.\
				add("cycles_cnt_en <= '0';")

		self.architecture.processes["cycles_cnt_en_gen"].\
				bodyHeader.add("wait for " + 
				str(3*clock_period)  + " ns;")

		self.architecture.processes["cycles_cnt_en_gen"].\
				bodyHeader.add("cycles_cnt_en <= '1';")

		self.architecture.processes["cycles_cnt_en_gen"].\
				bodyHeader.add("wait for " + 
				str(3*clock_period*self.n_cycles*2) \
				+ " ns;")

		self.architecture.processes["cycles_cnt_en_gen"].bodyHeader.\
				add("cycles_cnt_en <= '0';")

		# Cycles counter reset

		self.architecture.processes["cycles_cnt_rst_n_gen"].\
			sensitivity_list.add("clk")
		self.architecture.processes["cycles_cnt_rst_n_gen"].\
			sensitivity_list.add("stop")

		cycles_stop_if = If()
		cycles_stop_if._if_.conditions.add("stop = '1'")
		cycles_stop_if._if_.body.add("cycles_cnt_rst_n <= '0';")
		cycles_stop_if._else_.body.add("cycles_cnt_rst_n <= '1';")

		self.architecture.processes["cycles_cnt_rst_n_gen"].\
			if_list.add()
		self.architecture.processes["cycles_cnt_rst_n_gen"].\
			if_list[0]._if_.conditions.add("clk'event")
		self.architecture.processes["cycles_cnt_rst_n_gen"].\
			if_list[0]._if_.conditions.add("clk = '1'", "and")
		self.architecture.processes["cycles_cnt_rst_n_gen"].\
			if_list[0]._if_.body.add(cycles_stop_if)

		self.architecture.processes["cycles_cnt_rst_n_gen"].\
			final_wait = False

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
