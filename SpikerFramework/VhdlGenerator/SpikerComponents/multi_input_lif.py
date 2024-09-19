from math import log2

from .multi_input import MultiInput
from .lif_neuron import LIFneuron
from .testbench import Testbench
from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2, random_binary

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If


class MultiInputLIF(VHDLblock):

	def __init__(self, n_exc_inputs = 2, n_inh_inputs = 2, bitwidth = 16,
			w_inh_bw = 5, w_exc_bw = 5, shift = 10, debug = False,
			debug_list = []):

		self.name = "multi_input_lif"

		self.n_exc_inputs 	= n_exc_inputs
		self.n_inh_inputs 	= n_inh_inputs

		self.exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))

		self.bitwidth	= bitwidth
		self.w_inh_bw	= w_inh_bw
		self.w_exc_bw	= w_exc_bw
		self.shift 	= shift

		self.spiker_pkg = SpikerPackage()

		self.multi_input = MultiInput(
			n_exc_inputs = n_exc_inputs,
			n_inh_inputs = n_inh_inputs,
			debug = debug,
			debug_list = debug_list
		)

		self.lif_neuron = LIFneuron(
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
			name		= "exc_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.exc_cnt_bitwidth))
		self.entity.generic.add(
			name		= "inh_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.inh_cnt_bitwidth))
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
			print("Inhibitory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)
			

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
			print("Excitatory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)

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
			name 		= "restart", 
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
			name 		= "mi_ready", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "out_spike",
			direction	= "out",
			port_type	= "std_logic")


		# Signals
		self.architecture.signal.add(
			name 		= "exc", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "inh", 
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "exc_spike", 
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "inh_spike", 
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "neuron_restart", 
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "neuron_ready",
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "exc_cnt_sig", 
			signal_type	= "std_logic_vector("
						"exc_cnt_bitwidth - 1 "
						"downto 0)")

		self.architecture.signal.add(
			name 		= "inh_cnt_sig", 
			signal_type	= "std_logic_vector("
						"inh_cnt_bitwidth - 1 "
						"downto 0)")


		

		# Components
		self.architecture.component.add(self.multi_input)
		self.architecture.component.add(self.lif_neuron)

		self.architecture.bodyCodeHeader.add("exc_cnt <= exc_cnt_sig;")
		self.architecture.bodyCodeHeader.add("inh_cnt <= inh_cnt_sig;")
		
		# Multi-input
		self.architecture.instances.add(self.multi_input,
				"mi")
		self.architecture.instances["mi"].generic_map()
		self.architecture.instances["mi"].port_map()
		self.architecture.instances["mi"].p_map.add("all_ready",
				"neuron_ready")
		self.architecture.instances["mi"].p_map.add("ready", "mi_ready")
		self.architecture.instances["mi"].p_map.add("exc_cnt",
				"exc_cnt_sig")
		self.architecture.instances["mi"].p_map.add("inh_cnt",
				"inh_cnt_sig")

		# LIF neuron
		self.architecture.instances.add(self.lif_neuron,
				"lif_neuron")
		self.architecture.instances["lif_neuron"].generic_map()
		self.architecture.instances["lif_neuron"].port_map()
		self.architecture.instances["lif_neuron"].p_map.add("restart",
				"neuron_restart")


		# Debug
		if debug:
			debug_component(self, debug_list)

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)

class MultiInputLIF_tb(Testbench):

	def __init__(self, clock_period = 20, file_output = False,
			output_dir = "output", file_input = False, 
			input_dir = "", input_signal_list = [], bitwidth = 16,
			w_inh_bw = 5, w_exc_bw = 5, shift = 10, n_exc_inputs =
			2, n_inh_inputs = 2, debug = False, debug_list = []):

		self.n_exc_inputs 	= n_exc_inputs
		self.n_inh_inputs 	= n_inh_inputs

		self.exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))

		self.bitwidth	= bitwidth
		self.w_inh_bw	= w_inh_bw
		self.w_exc_bw	= w_exc_bw
		self.shift 	= shift

		self.spiker_pkg = SpikerPackage()

		self.dut = MultiInputLIF(
			bitwidth = bitwidth,
			w_inh_bw = w_inh_bw,
			w_exc_bw = w_exc_bw,
			shift = shift,
			n_exc_inputs = n_exc_inputs,
			n_inh_inputs = n_inh_inputs,
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
				add("exc_weight <= to_signed(500, "
				"exc_weight'length);")

		# inh_weight
		self.architecture.processes["inh_weight_gen"].bodyHeader.\
				add("inh_weight <= to_signed(-300, "
				"inh_weight'length);")

		# v_reset
		self.architecture.processes["v_reset_gen"].bodyHeader.\
				add("v_reset <= to_signed(1000, "
				"v_reset'length);")

		# v_th
		self.architecture.processes["v_th_gen"].bodyHeader.\
				add("v_th <= to_signed(3000, "
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
		mi_ready_if = If()
		mi_ready_if._if_.conditions.add("mi_ready = '1'")
		mi_ready_if._if_.body.add("start <= '1';")
		mi_ready_if._else_.body.add("start <= '0';")


		self.architecture.processes["start_gen"].final_wait = False
		self.architecture.processes["start_gen"].sensitivity_list.\
			add("clk")
		self.architecture.processes["start_gen"].if_list.add()
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk'event")
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk = '1'", "and")
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			body.add(mi_ready_if)

		# Exc spikes
		mi_ready_if = If()
		mi_ready_if._if_.conditions.add("start = '1'")
		mi_ready_if._if_.conditions.add("mi_ready = '1'", "and")
		mi_ready_if._if_.body.add("spikes_value := spikes_value + 1;")
		mi_ready_if._if_.body.add("exc_value := spikes_value;")
		mi_ready_if._elsif_.add()
		mi_ready_if._elsif_[0].conditions.add("start = '0'")
		mi_ready_if._elsif_[0].body.add("exc_value := 0;")


		self.architecture.processes["exc_spikes_gen"].\
				final_wait = False
		self.architecture.processes["exc_spikes_gen"].\
				sensitivity_list.add("clk")
		self.architecture.processes["exc_spikes_gen"].variables.add(
				name		= "spikes_value",
				var_type	= "integer",
				value		= "0")
		self.architecture.processes["exc_spikes_gen"].variables.add(
				name		= "exc_value",
				var_type	= "integer",
				value		= "0"
		)
		self.architecture.processes["exc_spikes_gen"].if_list.add()
		self.architecture.processes["exc_spikes_gen"].if_list[0].\
			_if_.conditions.add("clk'event")
		self.architecture.processes["exc_spikes_gen"].if_list[0].\
			_if_.conditions.add("clk = '1'", "and")
		self.architecture.processes["exc_spikes_gen"].if_list[0].\
			_if_.body.add(mi_ready_if)
		self.architecture.processes["exc_spikes_gen"].if_list[0].\
			_if_.body.add("exc_spikes <= std_logic_vector("
			"to_unsigned(exc_value, exc_spikes'length));")

		# Inh spikes
		mi_ready_if = If()
		mi_ready_if._if_.conditions.add("start = '1'")
		mi_ready_if._if_.conditions.add("mi_ready = '1'", "and")
		mi_ready_if._if_.body.add("spikes_value := spikes_value + 1;")
		mi_ready_if._if_.body.add("inh_value := spikes_value;")
		mi_ready_if._elsif_.add()
		mi_ready_if._elsif_[0].conditions.add("start = '0'")
		mi_ready_if._elsif_[0].body.add("inh_value := 0;")


		self.architecture.processes["inh_spikes_gen"].\
				final_wait = False
		self.architecture.processes["inh_spikes_gen"].\
				sensitivity_list.add("clk")
		self.architecture.processes["inh_spikes_gen"].variables.add(
				name		= "spikes_value",
				var_type	= "integer",
				value		= "0")
		self.architecture.processes["inh_spikes_gen"].variables.add(
				name		= "inh_value",
				var_type	= "integer",
				value		= "0"
		)
		self.architecture.processes["inh_spikes_gen"].if_list.add()
		self.architecture.processes["inh_spikes_gen"].if_list[0].\
			_if_.conditions.add("clk'event")
		self.architecture.processes["inh_spikes_gen"].if_list[0].\
			_if_.conditions.add("clk = '1'", "and")
		self.architecture.processes["inh_spikes_gen"].if_list[0].\
			_if_.body.add(mi_ready_if)
		self.architecture.processes["inh_spikes_gen"].if_list[0].\
			_if_.body.add("inh_spikes <= std_logic_vector("
			"to_unsigned(inh_value, inh_spikes'length));")

		# restart
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '0';")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"wait for 1500 ns;")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '1';")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"wait for 20 ns;")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '0';")
