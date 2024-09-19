from .lif_neuron_dp import LIFneuronDP
from .lif_neuron_cu import LIFneuronCU
from .and_mask import AndMask
from .testbench import Testbench

from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all

from .vhdltools.vhdl_block import VHDLblock

class LIFneuron(VHDLblock):

	def __init__(self, bitwidth = 16, w_inh_bw = 16, w_exc_bw = 16,
			shift = 10, reset = "fixed", debug = False, 
			debug_list = []):

		self.reset_types = [
			"fixed",
			"subtractive"
		]
		
		if reset not in self.reset_types:
			raise ValueError(str(reset) + " reset type not "
					"allowed")

		self.reset = reset

		self.name = "neuron"
		self.spiker_pkg = SpikerPackage()

		self.datapath = LIFneuronDP(
			bitwidth 	= bitwidth,
			w_inh_bw 	= w_inh_bw,
			w_exc_bw 	= w_exc_bw, 
			shift 		= shift,
			reset		= reset,
			debug		= debug,
			debug_list 	= debug_list
		)

		self.control_unit = LIFneuronCU(
			reset		= reset,
			debug 		= debug,
			debug_list 	= debug_list
		)
		
		self.and_mask = AndMask(data_type = "signed")

		self.bitwidth = bitwidth
		self.w_exc_bw = w_exc_bw
		self.w_inh_bw = w_inh_bw
		self.shift = shift
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

		if self.reset == "fixed":
			self.entity.port.add(
				name 		= "v_reset", 
				direction	= "in",
				port_type	= "signed(neuron_bit_width-1"
						"downto 0)")

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
			name 		= "restart", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "exc", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "inh", 
			direction	= "in", 
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "exc_spike", 
			direction	= "in", 
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "inh_spike", 
			direction	= "in", 
			port_type	= "std_logic")

		# Output
		self.entity.port.add(
			name 		= "neuron_ready",
			direction	= "out",
			port_type	= "std_logic")
		
		self.entity.port.add(
			name 		= "out_spike",
			direction	= "out",
			port_type	= "std_logic")

		# Signals
		self.architecture.signal.add(
			name 		= "update_sel",
			signal_type	= "std_logic_vector(1 downto 0)")
		self.architecture.signal.add(
			name 		= "add_or_sub", 
			signal_type	= "std_logic")

		if self.reset == "fixed":
			self.architecture.signal.add(
				name 		= "v_update",
				signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "v_en",
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "v_rst_n",
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "exceed_v_th",
			signal_type	= "std_logic")

		if self.w_inh_bw < self.bitwidth:
			self.architecture.signal.add(
				name 		= "masked_inh_weight",
				signal_type	= "signed("
						"inh_weights_bit_width-1 "
						"downto 0)")
		elif self.w_inh_bw == self.bitwidth:
			self.architecture.signal.add(
				name 		= "masked_inh_weight",
				signal_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			raise ValueError("Inhibitory weight bit-width cannot "
				"be larger than the neuron's one")
			

		if self.w_exc_bw < self.bitwidth:
			self.architecture.signal.add(
				name 		= "masked_exc_weight",
				signal_type	= "signed("
					"exc_weights_bit_width-1 downto 0)")

		elif self.w_exc_bw == self.bitwidth:
			self.architecture.signal.add(
				name 		= "masked_exc_weight",
				signal_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			raise ValueError("Excitatory weight bit-width cannot "
					"be larger than the neuron's one")


		# Components
		self.architecture.component.add(self.datapath)
		self.architecture.component.add(self.control_unit)
		self.architecture.component.add(self.and_mask)

		# Datapath
		self.architecture.instances.add(self.datapath,
				"datapath")
		self.architecture.instances["datapath"].generic_map()
		self.architecture.instances["datapath"].port_map()
		self.architecture.instances["datapath"].p_map.add("exc_weight",
				"masked_exc_weight")
		self.architecture.instances["datapath"].p_map.add("inh_weight",
				"masked_inh_weight")

		# Control unit
		self.architecture.instances.add(self.control_unit,
				"control_unit")
		self.architecture.instances["control_unit"].generic_map()
		self.architecture.instances["control_unit"].port_map()

		# Excitatory weights mask
		self.architecture.instances.add(self.and_mask,
				"exc_mask")
		self.architecture.instances["exc_mask"].generic_map()
		self.architecture.instances["exc_mask"].port_map()

		if self.w_inh_bw < self.bitwidth:
			self.architecture.instances["exc_mask"].g_map.add("N",
				"exc_weights_bit_width")

		elif self.w_inh_bw == self.bitwidth:
			self.architecture.instances["exc_mask"].g_map.add("N",
				"neuron_bit_width")
		else:
			raise ValueError("Excitatory weight bit-width cannot "
					"be larger than the neuron's one")

		self.architecture.instances["exc_mask"].p_map.add("input_bits",
				"exc_weight")
		self.architecture.instances["exc_mask"].p_map.add("mask_bit",
				"exc_spike")
		self.architecture.instances["exc_mask"].p_map.add("output_bits",
				"masked_exc_weight")

		# Inhibitory weights mask
		self.architecture.instances.add(self.and_mask,
				"inh_mask")
		self.architecture.instances["inh_mask"].generic_map()
		self.architecture.instances["inh_mask"].port_map()

		if self.w_inh_bw < self.bitwidth:
			self.architecture.instances["inh_mask"].g_map.add("N",
				"inh_weights_bit_width")

		elif self.w_inh_bw == self.bitwidth:
			self.architecture.instances["inh_mask"].g_map.add("N",
				"neuron_bit_width")
		else:
			raise ValueError("Inhibitory weight bit-width cannot "
				"be larger than the neuron's one")

		self.architecture.instances["inh_mask"].p_map.add("input_bits",
				"inh_weight")
		self.architecture.instances["inh_mask"].p_map.add("mask_bit",
				"inh_spike")
		self.architecture.instances["inh_mask"].p_map.add("output_bits",
				"masked_inh_weight")


		# Debug
		if debug:
			debug_component(self, debug_list)

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
	


class LIFneuron_tb(Testbench):

	def __init__(self, clock_period = 20, file_output = False, output_dir =
			"output", file_input = False, input_dir = "",
			input_signal_list = [], bitwidth = 16, w_inh_bw = 16,
			w_exc_bw = 16, shift = 10, reset = "fixed", 
			debug = False, debug_list = []):

		self.spiker_pkg = SpikerPackage()

		self.dut = LIFneuron(
			bitwidth = bitwidth,
			w_inh_bw = w_inh_bw,
			w_exc_bw = w_exc_bw,
			shift = shift,
			reset = reset,
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
		if self.dut.reset == "fixed":
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

		# restart
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '0';")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"wait for 70 ns;")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '1';")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"wait for 20 ns;")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '0';")

		# exc
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '0';")
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"wait for 130 ns;")
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '1';")
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"wait for 600 ns;")
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '0';")
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"wait for 100 ns;")
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '1';")
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"wait for 60 ns;")
		self.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '0';")

		# exc spike
		self.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"exc_spike <= '0';")
		self.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"wait for 130 ns;")
		self.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"exc_spike <= '1';")
		self.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"wait for 600 ns;")
		self.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"exc_spike <= '0';")

		# inh
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '0';")
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"wait for 750 ns;")
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '1';")
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"wait for 60 ns;")
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '0';")
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"wait for 20 ns;")
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '1';")
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"wait for 60 ns;")
		self.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '0';")

		# inh spike
		self.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"inh_spike <= '0';")
		self.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"wait for 750 ns;")
		self.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"inh_spike <= '1';")
		self.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"wait for 60 ns;")
		self.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"inh_spike <= '0';")
