import numpy as np

from math import log2

from .multi_cycle_lif import MultiCycleLIF
from .rom import Rom
from .addr_converter import AddrConverter
from .testbench import Testbench
from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2, random_binary

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If


class SingleLifBram(VHDLblock):

	def __init__(self,  n_exc_inputs = 2, n_inh_inputs = 2, n_cycles = 10,
			bitwidth = 16, w_inh_bw = 5, w_exc_bw = 5, shift = 10,
			w_exc_array = np.array([[0.1, 0.2]]), w_inh_array =
			np.array([[-0.1, -0.2]]), debug = False, 
			debug_list = []):

		self.name = "single_lif_bram"

		self.n_exc_inputs 	= w_exc_array.shape[1]
		self.n_inh_inputs 	= w_inh_array.shape[1]
		self.n_cycles		= n_cycles

		self.exc_cnt_bitwidth = int(log2(ceil_pow2(self.n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(self.n_inh_inputs)))
		self.cycles_cnt_bitwidth = int(log2(ceil_pow2(
			self.n_cycles+1))) + 1

		self.bitwidth 		= bitwidth
		self.w_exc_bw		= w_exc_bw
		self.w_inh_bw		= w_inh_bw
		self.shift		= shift

		self.spiker_pkg = SpikerPackage()

		self.lif_neuron = MultiCycleLIF(
			n_exc_inputs = self.n_exc_inputs,
			n_inh_inputs = self.n_inh_inputs,
			n_cycles = self.n_cycles,
			bitwidth = self.bitwidth,
			w_inh_bw = self.w_inh_bw,
			w_exc_bw = self.w_exc_bw,
			shift = self.shift,
			debug = debug,
			debug_list = debug_list)

		self.exc_mem = Rom(
			init_array 	= w_exc_array,
			bitwidth 	= w_exc_bw,
			name_term 	= "_exc"
		) 

		self.inh_mem = Rom(
			init_array 	= w_inh_array,
			bitwidth 	= w_inh_bw,
			name_term 	= "_inh"
		) 

		self.addr_converter = AddrConverter(
			bitwidth	= self.exc_cnt_bitwidth
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

		if self.w_inh_bw < self.bitwidth:
			self.architecture.signal.add(
				name 		= "inh_weight",
				signal_type	= "std_logic_vector("
						"inh_weights_bit_width-1 "
						"downto 0)")
		elif self.w_inh_bw == self.bitwidth:
			self.architecture.signal.add(
				name 		= "inh_weight",
				signal_type	= "std_logic_vector("
						"neuron_bit_width-1 "
						"downto 0)")
		else:
			print("Inhibitory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)
			

		if self.w_exc_bw < self.bitwidth:
			self.architecture.signal.add(
				name 		= "exc_weight",
				signal_type	= "std_logic_vector("
					"exc_weights_bit_width-1 downto 0)")

		elif self.w_exc_bw == self.bitwidth:
			self.architecture.signal.add(
				name 		= "exc_weight",
				signal_type	= "std_logic_vector("
						"neuron_bit_width-1 "
						"downto 0)")

		else:
			print("Excitatory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)

		self.architecture.signal.add(
			name 		= "exc_cnt", 
			signal_type	= "std_logic_vector("
						"exc_cnt_bitwidth - 1 "
						"downto 0)")
		self.architecture.signal.add(
			name 		= "inh_cnt", 
			signal_type	= "std_logic_vector("
						"inh_cnt_bitwidth - 1 "
						"downto 0)")

		self.architecture.signal.add(
			name		= "exc_addr",
			signal_type	= "std_logic_vector("
						"exc_cnt_bitwidth - 1 "
						"downto 0)"
		)

		self.architecture.signal.add(
			name		= "inh_addr",
			signal_type	= "std_logic_vector("
						"inh_cnt_bitwidth - 1 "
						"downto 0)"
		)

		# Components
		self.architecture.component.add(self.lif_neuron)
		self.architecture.component.add(self.exc_mem)
		self.architecture.component.add(self.inh_mem)
		self.architecture.component.add(self.addr_converter)

		# Multi-cycle
		self.architecture.instances.add(self.lif_neuron,
				"mc_lif")
		self.architecture.instances["mc_lif"].generic_map()
		self.architecture.instances["mc_lif"].port_map()
		self.architecture.instances["mc_lif"].p_map.add(
				"exc_weight", "signed(exc_weight)")
		self.architecture.instances["mc_lif"].p_map.add(
				"inh_weight", "signed(inh_weight)")

		# Excitatory memory
		self.architecture.instances.add(self.exc_mem,
				"exc_mem")
		self.architecture.instances["exc_mem"].generic_map()
		self.architecture.instances["exc_mem"].port_map()
		self.architecture.instances["exc_mem"].p_map.add(
				"dout_0", "exc_weight")
		self.architecture.instances["exc_mem"].p_map.add(
				"addra", "exc_addr")
		self.architecture.instances["exc_mem"].p_map.add(
				"clka", "clk")

		# Excitatory address converter
		self.architecture.instances.add(self.addr_converter,
				"exc_addr_conv")
		self.architecture.instances["exc_addr_conv"].generic_map()
		self.architecture.instances["exc_addr_conv"].g_map.add(
				"N", "exc_cnt_bitwidth")
		self.architecture.instances["exc_addr_conv"].port_map()
		self.architecture.instances["exc_addr_conv"].p_map.add(
				"addr_in", "exc_cnt")
		self.architecture.instances["exc_addr_conv"].p_map.add(
				"addr_out", "exc_addr")
	

		# Inhibitory memory
		self.architecture.instances.add(self.inh_mem,
				"inh_mem")
		self.architecture.instances["inh_mem"].generic_map()
		self.architecture.instances["inh_mem"].port_map()
		self.architecture.instances["inh_mem"].p_map.add(
				"dout_0", "inh_weight")
		self.architecture.instances["inh_mem"].p_map.add(
				"addra", "inh_addr")
		self.architecture.instances["inh_mem"].p_map.add(
				"clka", "clk")

		# Inhibitory address converter
		self.architecture.instances.add(self.addr_converter,
				"inh_addr_conv")
		self.architecture.instances["inh_addr_conv"].generic_map()
		self.architecture.instances["inh_addr_conv"].g_map.add(
				"N", "inh_cnt_bitwidth")
		self.architecture.instances["inh_addr_conv"].port_map()
		self.architecture.instances["inh_addr_conv"].p_map.add(
				"addr_in", "inh_cnt")
		self.architecture.instances["inh_addr_conv"].p_map.add(
				"addr_out", "inh_addr")

		# Debug
		if debug:
			debug_component(self, debug_list)

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
	


class SingleLifBram_tb(Testbench):

	def __init__(self, clock_period = 20, file_output = False,
			output_dir = "output", file_input = False, 
			input_dir = "", input_signal_list = [], 
			n_cycles = 10, bitwidth = 16, w_inh_bw = 5, w_exc_bw =
			5, shift = 10, w_exc_array = np.array([[0.1, 0.2]]),
			w_inh_array = np.array([[-0.1, -0.2]]), debug = False, 
			debug_list = []):

		self.n_exc_inputs 	= w_exc_array.shape[1]
		self.n_inh_inputs 	= w_inh_array.shape[1]
		self.n_cycles		= n_cycles

		self.exc_cnt_bitwidth = int(log2(ceil_pow2(self.n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(self.n_inh_inputs)))
		self.cycles_cnt_bitwidth = int(log2(ceil_pow2(self.n_cycles+1))) + 1

		self.bitwidth	= bitwidth
		self.w_inh_bw	= w_inh_bw
		self.w_exc_bw	= w_exc_bw
		self.shift 	= shift

		self.spiker_pkg = SpikerPackage()

		self.dut = SingleLifBram(
			bitwidth = bitwidth,
			w_inh_bw = w_inh_bw,
			w_exc_bw = w_exc_bw,
			w_inh_array = w_inh_array,
			w_exc_array = w_exc_array,
			shift = shift,
			n_exc_inputs = self.n_exc_inputs,
			n_inh_inputs = self.n_inh_inputs,
			n_cycles = self.n_cycles,
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
		
		self.vhdl(
			clock_period		= clock_period,
			file_output		= file_output,
			output_dir		= output_dir,
			file_input		= file_input,
			input_dir		= input_dir,
			input_signal_list 	= input_signal_list
			)


	def vhdl(self, clock_period = 20, file_output = False, output_dir =
			"output", file_input = False, input_dir = "",
			input_signal_list = []):

		self.library.add("work")
		self.library["work"].package.add("spiker_pkg")

		# v_reset
		self.architecture.processes["v_reset_gen"].bodyHeader.\
				add("v_reset <= to_signed(1000000, "
				"v_reset'length);")

		# v_th
		self.architecture.processes["v_th_gen"].bodyHeader.\
				add("v_th <= to_signed(8, "
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

		del self.architecture.processes["exc_spikes_rd_en_gen"]
		self.architecture.bodyCodeHeader.add("exc_spikes_rd_en <= "
				"start_all;")
		del self.architecture.processes["inh_spikes_rd_en_gen"]
		self.architecture.bodyCodeHeader.add("inh_spikes_rd_en <= "
				"start_all;")

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)


if __name__ == "__main__":

	from .vhdl import fast_compile, elaborate
	from .utils import generate_spikes

	generate_spikes("exc_spikes.txt", 10, 20)
	generate_spikes("inh_spikes.txt", 10, 20)

	tb = SingleLifBram_tb(
		n_cycles	= 20,
		bitwidth	= 16,
		w_exc_bw	= 16,
		w_inh_bw	= 16,
		shift		= 5,
		w_exc_array	= np.array([[0.1, 0.2, 0.8, 0.5, 0.3, 0.1, 0.4, 0.8,
			0.1, 0.3]]),
		w_inh_array	= np.array([[0.1, 0.2, 0.8, 0.5, 0.3, 0.1, 0.4, 0.8,
			0.3, 0.2]]),
		file_input = True,
		input_signal_list = [
			"exc_spikes",
			"inh_spikes"
		],
		debug = True,
		debug_list = [
			"single_lif_bram_exc_cnt",
			"single_lif_bram_exc_weight",
			"multi_input_lif_exc_spike",
			"neuron_cu_present_state",
			"neuron_datapath_v"
		]
	)

	print(tb.components)

	tb.write_file_all(rm = True)
	fast_compile(tb)
	elaborate(tb)
