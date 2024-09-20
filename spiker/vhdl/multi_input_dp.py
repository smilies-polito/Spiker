from math import log2

from .vhdl_or import Or
from .mux import Mux
from .reg import Reg
from .cnt import Cnt
from .cmp import Cmp
from .testbench import Testbench
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2, random_binary 

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If

class MultiInputDP(VHDLblock):

	def __init__(self, n_exc_inputs = 2, n_inh_inputs = 2, 
			debug = False, debug_list = []):

		self.name = "multi_input_dp_" + str(n_exc_inputs) + "_exc_" + \
			str(n_inh_inputs) + "_inh"

		self.n_exc_inputs = n_exc_inputs
		self.n_inh_inputs = n_inh_inputs
		self.exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))

		self.spikes_or = Or(bitwidth = n_exc_inputs)

		self.reg = Reg(
			bitwidth = n_exc_inputs, 
			reg_type = "std_logic_vector",
			rst = "sync", 
			active = "low"
		)

		if ceil_pow2(n_exc_inputs) == ceil_pow2(n_inh_inputs):
			self.mux = Mux(n_in = n_exc_inputs, 
					in_type = "std_logic", bitwidth = 1)
		else:
			self.exc_mux = Mux(n_in = n_exc_inputs, 
					in_type = "std_logic", bitwidth = 1)
			self.inh_mux = Mux(n_in = n_inh_inputs, 
					in_type = "std_logic", bitwidth = 1)

		self.counter = Cnt(bitwidth = self.exc_cnt_bitwidth) 

		self.cmp = Cmp(
			bitwidth = self.exc_cnt_bitwidth,
			cmp_type = "eq",
			signal_type = "std_logic"
		)

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
			name		= "n_exc_inputs", 
			gen_type	= "integer",
			value		= str(self.n_exc_inputs))
		self.entity.generic.add(
			name		= "n_inh_inputs", 
			gen_type	= "integer",
			value		= str(self.n_inh_inputs))
		self.entity.generic.add(
			name		= "exc_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.exc_cnt_bitwidth))
		self.entity.generic.add(
			name		= "inh_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.inh_cnt_bitwidth))


		self.entity.port.add(
			name 		= "clk", 
			direction	= "in", 
			port_type	= "std_logic")

		# Input spikes
		self.entity.port.add(
			name 		= "exc_spikes", 
			direction	= "in",
			port_type	= "std_logic_vector(n_exc_inputs-1 " 
						"downto 0)")
		self.entity.port.add(
			name 		= "inh_spikes", 
			direction	= "in",
			port_type	= "std_logic_vector(n_inh_inputs-1 " 
						"downto 0)")

		# Input controls
		self.entity.port.add(
			name 		= "exc_sample", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "exc_rst_n", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "exc_cnt_en", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "exc_cnt_rst_n", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "inh_sample", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "inh_rst_n", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "inh_cnt_en", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "inh_cnt_rst_n", 
			direction	= "in", 
			port_type	= "std_logic")

		# Output
		self.entity.port.add(
			name 		= "exc_yes", 
			direction	= "out", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "exc_spike", 
			direction	= "out", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "exc_stop", 
			direction	= "out", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "exc_cnt", 
			direction	= "out", 
			port_type	= "std_logic_vector("
						"exc_cnt_bitwidth - 1 "
						"downto 0)")
		self.entity.port.add(
			name 		= "inh_yes", 
			direction	= "out", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "inh_spike", 
			direction	= "out", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "inh_stop", 
			direction	= "out", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "inh_cnt", 
			direction	= "out", 
			port_type	= "std_logic_vector("
						"inh_cnt_bitwidth - 1 "
						"downto 0)")

		# Signals
		self.architecture.signal.add(
			name		= "exc_spikes_sampled",
			signal_type	= "std_logic_vector(n_exc_inputs-1 "
						"downto 0)")
		self.architecture.signal.add(
			name		= "inh_spikes_sampled",
			signal_type	= "std_logic_vector(n_inh_inputs-1 "
						"downto 0)")
		self.architecture.signal.add(
			name		= "exc_cnt_sig",
			signal_type	= "std_logic_vector(exc_cnt_bitwidth-1 "
						"downto 0)")
		self.architecture.signal.add(
			name		= "inh_cnt_sig",
			signal_type	= "std_logic_vector(inh_cnt_bitwidth-1 "
						"downto 0)")


		# Components
		self.architecture.component.add(self.spikes_or)
		self.architecture.component.add(self.reg)
		self.architecture.component.add(self.counter)
		self.architecture.component.add(self.cmp)

		if ceil_pow2(self.n_exc_inputs) == ceil_pow2(self.n_inh_inputs):
			self.architecture.component.add(self.mux)
		else:
			self.architecture.component.add(self.exc_mux)
			self.architecture.component.add(self.inh_mux)
			

		# Connect output signals
		self.architecture.bodyCodeHeader.add("exc_cnt <= exc_cnt_sig;")
		self.architecture.bodyCodeHeader.add("inh_cnt <= inh_cnt_sig;")


		# Exc or
		self.architecture.instances.add(self.spikes_or, "exc_or")
		self.architecture.instances["exc_or"].generic_map()
		self.architecture.instances["exc_or"].g_map.add("N",
				"n_exc_inputs")
		self.architecture.instances["exc_or"].port_map()
		self.architecture.instances["exc_or"].p_map.add("or_in",
				"exc_spikes")
		self.architecture.instances["exc_or"].p_map.add("or_out",
				"exc_yes")

		# Inh or
		self.architecture.instances.add(self.spikes_or, "inh_or")
		self.architecture.instances["inh_or"].generic_map()
		self.architecture.instances["inh_or"].g_map.add("N",
				"n_inh_inputs")
		self.architecture.instances["inh_or"].port_map()
		self.architecture.instances["inh_or"].p_map.add("or_in",
				"inh_spikes")
		self.architecture.instances["inh_or"].p_map.add("or_out",
				"inh_yes")

		# Exc reg
		self.architecture.instances.add(self.reg, "exc_reg")
		self.architecture.instances["exc_reg"].generic_map()
		self.architecture.instances["exc_reg"].g_map.add("N",
				"n_exc_inputs")
		self.architecture.instances["exc_reg"].port_map()
		self.architecture.instances["exc_reg"].p_map.add("reg_in",
				"exc_spikes")
		self.architecture.instances["exc_reg"].p_map.add("en",
				"exc_sample")
		self.architecture.instances["exc_reg"].p_map.add("rst_n",
				"exc_rst_n")
		self.architecture.instances["exc_reg"].p_map.add("reg_out",
				"exc_spikes_sampled")

		# Inh reg
		self.architecture.instances.add(self.reg, "inh_reg")
		self.architecture.instances["inh_reg"].generic_map()
		self.architecture.instances["inh_reg"].g_map.add("N",
				"n_inh_inputs")
		self.architecture.instances["inh_reg"].port_map()
		self.architecture.instances["inh_reg"].p_map.add("reg_in",
				"inh_spikes")
		self.architecture.instances["inh_reg"].p_map.add("en",
				"inh_sample")
		self.architecture.instances["inh_reg"].p_map.add("rst_n",
				"inh_rst_n")
		self.architecture.instances["inh_reg"].p_map.add("reg_out",
				"inh_spikes_sampled")



		if ceil_pow2(self.n_exc_inputs) == ceil_pow2(self.n_inh_inputs):

			# Exc multiplexer
			self.architecture.instances.add(self.mux, "exc_mux")
			self.architecture.instances["exc_mux"].port_map()

			if self.mux.entity.port["mux_sel"].port_type == \
				"std_logic":

				self.architecture.instances["exc_mux"].p_map.\
						add( "mux_sel", 
						"exc_cnt_sig(0)")
			else:
				self.architecture.instances["exc_mux"].p_map.\
						add( "mux_sel", "exc_cnt_sig")

			for i in range(ceil_pow2(self.n_exc_inputs)):
				if i < self.n_exc_inputs:
					self.architecture.instances["exc_mux"].\
						p_map.add("in" + str(i), 
						"exc_spikes_sampled(" + str(i) \
						+ ")")
				else:
					self.architecture.instances["exc_mux"].\
						p_map.add("in" + str(i), 
						"\'0\'")

			self.architecture.instances["exc_mux"].p_map.add(
					"mux_out", "exc_spike")

			# Inh multiplexer
			self.architecture.instances.add(self.mux, "inh_mux")
			self.architecture.instances["inh_mux"].port_map()

			if self.mux.entity.port["mux_sel"].port_type == \
				"std_logic":

				self.architecture.instances["inh_mux"].p_map.\
						add("mux_sel", 
						"inh_cnt_sig(0)")
			else:
				self.architecture.instances["inh_mux"].p_map.\
						add("mux_sel", "inh_cnt_sig")

			for i in range(ceil_pow2(self.n_inh_inputs)):
				if i < self.n_inh_inputs:
					self.architecture.instances["inh_mux"].\
						p_map.add("in" + str(i), 
						"inh_spikes_sampled(" + str(i) \
						+ ")")
				else:
					self.architecture.instances["inh_mux"].\
						p_map.add("in" + str(i), 
						"\'0\'")

			self.architecture.instances["inh_mux"].p_map.add(
					"mux_out", "inh_spike")
		else:
			# Exc multiplexer
			self.architecture.instances.add(self.exc_mux, "exc_mux")
			self.architecture.instances["exc_mux"].port_map()

			if self.exc_mux.entity.port["mux_sel"].port_type == \
				"std_logic":

				self.architecture.instances["exc_mux"].p_map.\
						add("mux_sel", "exc_cnt_sig(0)")
			else:
				self.architecture.instances["exc_mux"].p_map.\
						add("mux_sel", "exc_cnt_sig")

			for i in range(ceil_pow2(self.n_exc_inputs)):
				if i < self.n_exc_inputs:
					self.architecture.instances["exc_mux"].\
						p_map.add("in" + str(i), 
						"exc_spikes_sampled(" + str(i) \
						+ ")")
				else:
					self.architecture.instances["exc_mux"].\
						p_map.add("in" + str(i), 
						"\'0\'")

			self.architecture.instances["exc_mux"].p_map.add(
					"mux_out", "exc_spike")


			# Inh multiplexer
			self.architecture.instances.add(self.inh_mux, "inh_mux")
			self.architecture.instances["inh_mux"].port_map()

			if self.inh_mux.entity.port["mux_sel"].port_type == \
				"std_logic":

				self.architecture.instances["inh_mux"].p_map.\
						add("mux_sel", "inh_cnt_sig(0)")
			else:
				self.architecture.instances["inh_mux"].p_map.\
						add("mux_sel", "inh_cnt_sig")

			for i in range(ceil_pow2(self.n_inh_inputs)):
				if i < self.n_inh_inputs:
					self.architecture.instances["inh_mux"].\
						p_map.add("in" + str(i), 
						"inh_spikes_sampled(" + str(i) \
						+ ")")
				else:
					self.architecture.instances["inh_mux"].\
						p_map.add("in" + str(i), 
						"\'0\'")

			self.architecture.instances["inh_mux"].p_map.add(
					"mux_out", "inh_spike")


		# Exc cnt
		self.architecture.instances.add(self.counter, "exc_counter")
		self.architecture.instances["exc_counter"].generic_map()
		self.architecture.instances["exc_counter"].g_map.add("N",
				"exc_cnt_bitwidth")
		self.architecture.instances["exc_counter"].g_map.add(
				"rst_value", str(2**self.exc_cnt_bitwidth-1))
		self.architecture.instances["exc_counter"].port_map()
		self.architecture.instances["exc_counter"].p_map.add(
				"cnt_en", "exc_cnt_en")
		self.architecture.instances["exc_counter"].p_map.add(
				"cnt_rst_n", "exc_cnt_rst_n")
		self.architecture.instances["exc_counter"].p_map.add("cnt_out",
				"exc_cnt_sig")

		# Inh cnt
		self.architecture.instances.add(self.counter, "inh_counter")
		self.architecture.instances["inh_counter"].generic_map()
		self.architecture.instances["inh_counter"].g_map.add("N",
				"inh_cnt_bitwidth")
		self.architecture.instances["inh_counter"].g_map.add(
				"rst_value", str(2**self.inh_cnt_bitwidth-1))
		self.architecture.instances["inh_counter"].port_map()
		self.architecture.instances["inh_counter"].p_map.add(
				"cnt_en", "inh_cnt_en")
		self.architecture.instances["inh_counter"].p_map.add(
				"cnt_rst_n", "inh_cnt_rst_n")
		self.architecture.instances["inh_counter"].p_map.add("cnt_out",
				"inh_cnt_sig")

		# Exc cmp
		self.architecture.instances.add(self.cmp, "exc_cmp")
		self.architecture.instances["exc_cmp"].generic_map()
		self.architecture.instances["exc_cmp"].g_map.add("N",
				"exc_cnt_bitwidth")
		self.architecture.instances["exc_cmp"].port_map()
		self.architecture.instances["exc_cmp"].p_map.add(
				"in0", "exc_cnt_sig")
		self.architecture.instances["exc_cmp"].p_map.add(
				"in1", "std_logic_vector(to_unsigned("
				"n_exc_inputs-2, exc_cnt_bitwidth))")
		self.architecture.instances["exc_cmp"].p_map.add("cmp_out",
				"exc_stop")

		# Inh cmp
		self.architecture.instances.add(self.cmp, "inh_cmp")
		self.architecture.instances["inh_cmp"].generic_map()
		self.architecture.instances["inh_cmp"].g_map.add("N",
				"inh_cnt_bitwidth")
		self.architecture.instances["inh_cmp"].port_map()
		self.architecture.instances["inh_cmp"].p_map.add(
				"in0", "inh_cnt_sig")
		self.architecture.instances["inh_cmp"].p_map.add(
				"in1", "std_logic_vector(to_unsigned("
				"n_inh_inputs-2, inh_cnt_bitwidth))")
		self.architecture.instances["inh_cmp"].p_map.add("cmp_out",
				"inh_stop")


		# Debug
		if debug:
			debug_component(self, debug_list)

	
	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)



class MultiCycleDP_tb(Testbench):

	def __init__(self, clock_period = 20, file_output = False, output_dir =
			"output", file_input = False, input_dir = "",
			input_signal_list = [], n_exc_inputs = 2,
			n_inh_inputs = 2, debug = False, debug_list = []):

		self.n_exc_inputs = n_exc_inputs
		self.n_inh_inputs = n_inh_inputs

		self.dut = MultiCycleDP(
			n_exc_inputs	= n_exc_inputs,
			n_inh_inputs	= n_inh_inputs
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
		
		self.vhdl(clock_period = clock_period, file_output = 
			file_output)

	def vhdl(self, clock_period = 20, file_output = False):

		self.architecture.processes["exc_spikes_gen"].\
				bodyHeader.add("wait for " +
				str(clock_period) + " ns;")

		# Excitatory spikes
		for i in range(3):

			exc_spikes = "\"" + random_binary(0, 
					2**self.n_exc_inputs-1, 
					self.n_exc_inputs) + "\""

			self.architecture.processes["exc_spikes_gen"].\
					bodyHeader.add("exc_spikes <= "
					+ exc_spikes + ";")

			self.architecture.processes["exc_spikes_gen"].\
					bodyHeader.add("wait for " +
					str(self.n_exc_inputs * 
					clock_period * 2) + " ns;")


		self.architecture.processes["inh_spikes_gen"].\
				bodyHeader.add("wait for " +
				str(clock_period) + " ns;")

		# Inhibitory spikes
		for i in range(3):

			inh_spikes = "\"" + random_binary(0, 
					2**self.n_inh_inputs-1, 
					self.n_inh_inputs) + "\""

			self.architecture.processes["inh_spikes_gen"].\
					bodyHeader.add("inh_spikes <= "
					+ inh_spikes + ";")

			self.architecture.processes["inh_spikes_gen"].\
					bodyHeader.add("wait for " +
					str(self.n_inh_inputs * 
					clock_period * 2) + " ns;")

		# Exc register reset
		self.architecture.processes["exc_rst_n_gen"].bodyHeader.\
				add("exc_rst_n <= '0';")

		self.architecture.processes["exc_rst_n_gen"].\
				bodyHeader.add("wait for " + 
				str(clock_period)  + " ns;")

		self.architecture.processes["exc_rst_n_gen"].\
				bodyHeader.add("exc_rst_n <= '1';")

		# Exc register sample
		self.architecture.processes["exc_sample_gen"].bodyHeader.\
				add("exc_sample <= '0';")


		self.architecture.processes["exc_sample_gen"].\
				bodyHeader.add("wait for " + 
				str(2*clock_period)  + " ns;")

		for i in range(3):
			self.architecture.processes["exc_sample_gen"].\
					bodyHeader.add("exc_sample <= '1';")
			self.architecture.processes["exc_sample_gen"].\
					bodyHeader.add("wait for " + 
					str(clock_period)  + " ns;")
			self.architecture.processes["exc_sample_gen"].\
					bodyHeader.add("exc_sample <= '0';")
			self.architecture.processes["exc_sample_gen"].\
					bodyHeader.add("wait for " + 
					str(self.n_exc_inputs*clock_period*2) \
					+ " ns;")

		# Inh register reset
		self.architecture.processes["inh_rst_n_gen"].bodyHeader.\
				add("inh_rst_n <= '0';")

		self.architecture.processes["inh_rst_n_gen"].\
				bodyHeader.add("wait for " + 
				str(clock_period)  + " ns;")

		self.architecture.processes["inh_rst_n_gen"].\
				bodyHeader.add("inh_rst_n <= '1';")

		# Inh register sample
		self.architecture.processes["inh_sample_gen"].bodyHeader.\
				add("inh_sample <= '0';")


		self.architecture.processes["inh_sample_gen"].\
				bodyHeader.add("wait for " + 
				str(2*clock_period)  + " ns;")

		for i in range(3):
			self.architecture.processes["inh_sample_gen"].\
					bodyHeader.add("inh_sample <= '1';")
			self.architecture.processes["inh_sample_gen"].\
					bodyHeader.add("wait for " + 
					str(clock_period)  + " ns;")
			self.architecture.processes["inh_sample_gen"].\
					bodyHeader.add("inh_sample <= '0';")
			self.architecture.processes["inh_sample_gen"].\
					bodyHeader.add("wait for " + 
					str(self.n_inh_inputs*clock_period*2) \
					+ " ns;")

		# Exc counter enable
		self.architecture.processes["exc_cnt_en_gen"].bodyHeader.\
				add("exc_cnt_en <= '0';")


		self.architecture.processes["exc_cnt_en_gen"].\
				bodyHeader.add("wait for " + 
				str(3*clock_period)  + " ns;")

		self.architecture.processes["exc_cnt_en_gen"].\
				bodyHeader.add("exc_cnt_en <= '1';")

		self.architecture.processes["exc_cnt_en_gen"].\
				bodyHeader.add("wait for " + 
				str(3*clock_period*self.n_exc_inputs*2) \
				+ " ns;")

		self.architecture.processes["exc_cnt_en_gen"].bodyHeader.\
				add("exc_cnt_en <= '0';")

		# Inh counter enable
		self.architecture.processes["inh_cnt_en_gen"].bodyHeader.\
				add("inh_cnt_en <= '0';")


		self.architecture.processes["inh_cnt_en_gen"].\
				bodyHeader.add("wait for " + 
				str(3*clock_period)  + " ns;")

		self.architecture.processes["inh_cnt_en_gen"].\
				bodyHeader.add("inh_cnt_en <= '1';")

		self.architecture.processes["inh_cnt_en_gen"].\
				bodyHeader.add("wait for " + 
				str(3*clock_period*self.n_inh_inputs*2) \
				+ " ns;")

		self.architecture.processes["inh_cnt_en_gen"].bodyHeader.\
				add("inh_cnt_en <= '0';")


		# Exc counter reset

		self.architecture.processes["exc_cnt_rst_n_gen"].\
			sensitivity_list.add("clk")
		self.architecture.processes["exc_cnt_rst_n_gen"].\
			sensitivity_list.add("exc_stop")

		exc_stop_if = If()
		exc_stop_if._if_.conditions.add("exc_stop = '1'")
		exc_stop_if._if_.body.add("exc_cnt_rst_n <= '0';")
		exc_stop_if._else_.body.add("exc_cnt_rst_n <= '1';")

		self.architecture.processes["exc_cnt_rst_n_gen"].if_list.add()
		self.architecture.processes["exc_cnt_rst_n_gen"].\
			if_list[0]._if_.conditions.add("clk'event")
		self.architecture.processes["exc_cnt_rst_n_gen"].\
			if_list[0]._if_.conditions.add("clk = '1'", "and")
		self.architecture.processes["exc_cnt_rst_n_gen"].\
			if_list[0]._if_.body.add(exc_stop_if)

		self.architecture.processes["exc_cnt_rst_n_gen"].\
			final_wait = False

		# Inh counter reset

		self.architecture.processes["inh_cnt_rst_n_gen"].\
			sensitivity_list.add("clk")
		self.architecture.processes["inh_cnt_rst_n_gen"].\
			sensitivity_list.add("inh_stop")

		inh_stop_if = If()
		inh_stop_if._if_.conditions.add("inh_stop = '1'")
		inh_stop_if._if_.body.add("inh_cnt_rst_n <= '0';")
		inh_stop_if._else_.body.add("inh_cnt_rst_n <= '1';")

		self.architecture.processes["inh_cnt_rst_n_gen"].if_list.add()
		self.architecture.processes["inh_cnt_rst_n_gen"].\
			if_list[0]._if_.conditions.add("clk'event")
		self.architecture.processes["inh_cnt_rst_n_gen"].\
			if_list[0]._if_.conditions.add("clk = '1'", "and")
		self.architecture.processes["inh_cnt_rst_n_gen"].\
			if_list[0]._if_.body.add(inh_stop_if)

		self.architecture.processes["inh_cnt_rst_n_gen"].\
			final_wait = False
