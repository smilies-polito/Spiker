from math import log2

from .multi_cycle import MultiCycle
from .multi_input_lif import MultiInputLIF
from .testbench import Testbench
from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2, random_binary

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If


class MultiCycleLIF(VHDLblock):

	def __init__(self, n_exc_inputs = 2, n_inh_inputs = 2, n_cycles = 10,
			bitwidth = 16, w_inh_bw = 5, w_exc_bw = 5, shift = 10,
			debug = False, debug_list = []):

		self.name = "multi_cycle_lif"

		self.n_exc_inputs 	= n_exc_inputs
		self.n_inh_inputs 	= n_inh_inputs
		self.n_cycles		= n_cycles

		self.exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))
		self.cycles_cnt_bitwidth = int(log2(ceil_pow2(n_cycles+1))) + 1

		self.bitwidth	= bitwidth
		self.w_inh_bw	= w_inh_bw
		self.w_exc_bw	= w_exc_bw
		self.shift 	= shift

		self.spiker_pkg = SpikerPackage()

		self.multi_cycle = MultiCycle(
			n_cycles = n_cycles,
			debug = debug,
			debug_list = debug_list
		)

		self.multi_input_lif = MultiInputLIF(
			n_exc_inputs = n_exc_inputs,
			n_inh_inputs = n_inh_inputs,
			bitwidth = bitwidth,
			w_inh_bw = w_inh_bw,
			w_exc_bw = w_exc_bw,
			shift = shift,
			debug = debug,
			debug_list = debug_list
		)

		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		self.library.add("work")
		self.library["work"].package.add("spiker_pkg")
				

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
			name		= "n_cycles", 
			gen_type	= "integer",
			value		= str(self.n_cycles))
		self.entity.generic.add(
			name		= "exc_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.exc_cnt_bitwidth))
		self.entity.generic.add(
			name		= "inh_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.inh_cnt_bitwidth))
		self.entity.generic.add(
			name		= "cycles_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.cycles_cnt_bitwidth))
		self.entity.generic.add(
			name		= "neuron_bit_width", 
			gen_type	= "integer",
			value		= str(self.bitwidth))


		if self.w_inh_bw < self.bitwidth:
			self.entity.generic.add(
				name		= "inh_weights_bit_width",
				gen_type	= "integer",
				value		= str(
					self.w_inh_bw))

		if self.w_exc_bw < self.bitwidth:
			self.entity.generic.add(
				name		= "exc_weights_bit_width",
				gen_type	= "integer",
				value		= str(
					self.w_exc_bw))

		self.entity.generic.add(
			name		= "shift",
			gen_type	= "integer",
			value		= str(self.shift))

		# Input parameters
		self.entity.port.add(
			name 		= "v_th", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")
		self.entity.port.add(
			name 		= "v_reset", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")


		if self.w_inh_bw < self.bitwidth:
			self.entity.port.add(
				name 		= "inh_weight",
				direction	= "in",
				port_type	= "signed("
						"inh_weights_bit_width-1 "
						"downto 0)")
		elif self.w_inh_bw == self.bitwidth:
			self.entity.port.add(
				name 		= "inh_weight",
				direction	= "in",
				port_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			raise ValueError("Inhibitory weight bit-width cannot "
				"be larger than the neuron's one")
			

		if self.w_exc_bw < self.bitwidth:
			self.entity.port.add(
				name 		= "exc_weight",
				direction	= "in",
				port_type	= "signed("
					"exc_weights_bit_width-1 downto 0)")

		elif self.w_exc_bw == self.bitwidth:
			self.entity.port.add(
				name 		= "exc_weight",
				direction	= "in",
				port_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			raise ValueError("Excitatory weight bit-width cannot "
				"be larger than the neuron's one")

		# Input controls
		self.entity.port.add(
			name 		= "clk", 
			direction	= "in", 
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "rst_n", 
			direction	= "in", 
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "start", 
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

		# Output
		self.entity.port.add(
			name 		= "exc_cnt", 
			direction	= "out", 
			port_type	= "std_logic_vector("
						"exc_cnt_bitwidth - 1 "
						"downto 0)")
		self.entity.port.add(
			name 		= "inh_cnt", 
			direction	= "out", 
			port_type	= "std_logic_vector("
						"inh_cnt_bitwidth - 1 "
						"downto 0)")
		self.entity.port.add(
			name 		= "mc_ready", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "out_spike",
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "start_all",
			direction	= "out",
			port_type	= "std_logic")


		# Signals
		self.architecture.signal.add(
			name 		= "start_all_sig", 
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "restart", 
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "all_ready",
			signal_type	= "std_logic")
		

		# Components
		self.architecture.component.add(self.multi_cycle)
		self.architecture.component.add(self.multi_input_lif)

		self.architecture.bodyCodeHeader.add("start_all <= "
				"start_all_sig;")
		
		# Multi-cycle
		self.architecture.instances.add(self.multi_cycle,
				"mc")
		self.architecture.instances["mc"].generic_map()
		self.architecture.instances["mc"].port_map()
		self.architecture.instances["mc"].p_map.add("ready", "mc_ready")
		self.architecture.instances["mc"].p_map.add("start_all",
				"start_all_sig")

		# Multi-input neuron
		self.architecture.instances.add(self.multi_input_lif,
				"multi_input_neuron")
		self.architecture.instances["multi_input_neuron"].generic_map()
		self.architecture.instances["multi_input_neuron"].port_map()
		self.architecture.instances["multi_input_neuron"].p_map.add(
				"start", "start_all_sig")
		self.architecture.instances["multi_input_neuron"].p_map.add(
				"mi_ready", "all_ready")


		# Debug
		if debug:
			debug_component(self, debug_list)

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)


class MultiCycleLIF_tb(Testbench):

	def __init__(self, clock_period = 20, file_output = False,
			output_dir = "output", file_input = False, 
			input_dir = "", input_signal_list = [], bitwidth = 16,
			w_inh_bw = 5, w_exc_bw = 5, shift = 10, n_exc_inputs =
			2, n_inh_inputs = 2, n_cycles = 10, debug = False, 
			debug_list = []):

		self.n_exc_inputs 	= n_exc_inputs
		self.n_inh_inputs 	= n_inh_inputs
		self.n_cycles		= n_cycles

		self.exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))
		self.cycles_cnt_bitwidth = int(log2(ceil_pow2(n_cycles+1))) + 1

		self.bitwidth	= bitwidth
		self.w_inh_bw	= w_inh_bw
		self.w_exc_bw	= w_exc_bw
		self.shift 	= shift

		self.spiker_pkg = SpikerPackage()

		self.dut = MultiCycleLIF(
			bitwidth = bitwidth,
			w_inh_bw = w_inh_bw,
			w_exc_bw = w_exc_bw,
			shift = shift,
			n_exc_inputs = n_exc_inputs,
			n_inh_inputs = n_inh_inputs,
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


	def vhdl(self, clock_period = 20, file_output = False, output_dir =
			"output", file_input = False, input_dir = "",
			input_signal_list = []):

		self.library.add("work")
		self.library["work"].package.add("spiker_pkg")

		# exc_weight
		self.architecture.processes["exc_weight_gen"].bodyHeader.\
				add("exc_weight <= to_signed(550000000, "
				"exc_weight'length);")

		# inh_weight
		self.architecture.processes["inh_weight_gen"].bodyHeader.\
				add("inh_weight <= to_signed(-400000000, "
				"inh_weight'length);")

		# v_reset
		self.architecture.processes["v_reset_gen"].bodyHeader.\
				add("v_reset <= to_signed(1000000, "
				"v_reset'length);")

		# v_th
		self.architecture.processes["v_th_gen"].bodyHeader.\
				add("v_th <= to_signed(1500000000, "
				"v_th'length);")

		# rst_n
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '1';")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"wait for 15 ns;")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '0';")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"wait for 10 ns;")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '1';")

		# Start
		mc_ready_if = If()
		mc_ready_if._if_.conditions.add("mc_ready = '1'")
		mc_ready_if._if_.body.add("start <= '1';")
		mc_ready_if._else_.body.add("start <= '0';")


		self.architecture.processes["start_gen"].final_wait = False
		self.architecture.processes["start_gen"].sensitivity_list.\
			add("clk")
		self.architecture.processes["start_gen"].if_list.add()
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk'event")
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk = '1'", "and")
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			body.add(mc_ready_if)


		del self.architecture.processes["exc_spikes_gen"]
		del self.architecture.processes["inh_spikes_gen"]
		self.load(signal_name = "exc_spikes", input_dir = input_dir)
		self.load(signal_name = "inh_spikes", input_dir = input_dir)

		del self.architecture.processes["exc_spikes_rd_en_gen"]
		self.architecture.bodyCodeHeader.add("exc_spikes_rd_en <= "
				"start_all;")
		del self.architecture.processes["inh_spikes_rd_en_gen"]
		self.architecture.bodyCodeHeader.add("inh_spikes_rd_en <= "
				"start_all;")
