import subprocess as sp
from math import log2

import path_config

from vhdl_block import VHDLblock
from if_statement import If
from multi_cycle import MultiCycle
from multi_input_lif import MultiInputLIF
from testbench import Testbench

from spiker_pkg import SpikerPackage
from utils import track_signals, ceil_pow2, debug_component


class MultiCycleLIF(VHDLblock):

	def __init__(self, n_exc_inputs = 2, n_inh_inputs = 2, n_cycles = 10,
			bitwidth = 16, w_inh_bw = 5, w_exc_bw = 5, shift = 10,
			debug = False, debug_list = []):

		self.n_exc_inputs 	= n_exc_inputs
		self.n_inh_inputs 	= n_inh_inputs
		self.n_cycles		= n_cycles

		exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))
		cycles_cnt_bitwidth = int(log2(ceil_pow2(n_cycles+1))) + 1

		VHDLblock.__init__(self, entity_name = "multi_cycle_lif")

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
			value		= str(n_exc_inputs))
		self.entity.generic.add(
			name		= "n_inh_inputs", 
			gen_type	= "integer",
			value		= str(n_inh_inputs))
		self.entity.generic.add(
			name		= "n_cycles", 
			gen_type	= "integer",
			value		= str(n_cycles))
		self.entity.generic.add(
			name		= "exc_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(exc_cnt_bitwidth))
		self.entity.generic.add(
			name		= "inh_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(inh_cnt_bitwidth))
		self.entity.generic.add(
			name		= "cycles_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(cycles_cnt_bitwidth))
		self.entity.generic.add(
			name		= "neuron_bit_width", 
			gen_type	= "integer",
			value		= str(bitwidth))


		if w_inh_bw < bitwidth:
			self.entity.generic.add(
				name		= "inh_weights_bit_width",
				gen_type	= "integer",
				value		= str(
					w_inh_bw))

		if w_exc_bw < bitwidth:
			self.entity.generic.add(
				name		= "exc_weights_bit_width",
				gen_type	= "integer",
				value		= str(
					w_exc_bw))

		self.entity.generic.add(
			name		= "shift",
			gen_type	= "integer",
			value		= str(shift))

		# Input parameters
		self.entity.port.add(
			name 		= "v_th_value", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")
		self.entity.port.add(
			name 		= "v_reset", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")


		if w_inh_bw < bitwidth:
			self.entity.port.add(
				name 		= "inh_weight",
				direction	= "in",
				port_type	= "signed("
						"inh_weights_bit_width-1 "
						"downto 0)")
		elif w_inh_bw == bitwidth:
			self.entity.port.add(
				name 		= "inh_weight",
				direction	= "in",
				port_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			print("Inhibitory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)
			

		if w_exc_bw < bitwidth:
			self.entity.port.add(
				name 		= "exc_weight",
				direction	= "in",
				port_type	= "signed("
					"exc_weights_bit_width-1 downto 0)")

		elif w_exc_bw == bitwidth:
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
			name 		= "v_th_en", 
			direction	= "in", 
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "neuron_load_end", 
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
			name 		= "neuron_load_ready",
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
	
		

	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")


	def compile_all(self, output_dir = "output"):

		self.spiker_pkg.compile(output_dir = output_dir)
		self.multi_cycle.compile_all(output_dir = output_dir)
		self.multi_input_lif.compile_all(output_dir = output_dir)

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")


	def write_file_all(self, output_dir = "output"):

		self.spiker_pkg.write_file(output_dir = output_dir)
		self.multi_cycle.write_file_all(output_dir = output_dir)
		self.multi_input_lif.write_file_all(output_dir = output_dir)
		self.write_file(output_dir = output_dir)



	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")


	def testbench(self, clock_period = 20, file_output = False,
			output_dir = "output", file_input = False, 
			input_dir = "", input_signal_list = []):

		self.tb = Testbench(
			self,
			clock_period		= clock_period,
			file_output		= file_output,
			output_dir		= output_dir,
			file_input		= file_input,
			input_signal_list	= input_signal_list
		)

		self.tb.library.add("work")
		self.tb.library["work"].package.add("spiker_pkg")

		# exc_weight
		self.tb.architecture.processes["exc_weight_gen"].bodyHeader.\
				add("exc_weight <= to_signed(550000000, "
				"exc_weight'length);")

		# inh_weight
		self.tb.architecture.processes["inh_weight_gen"].bodyHeader.\
				add("inh_weight <= to_signed(-400000000, "
				"inh_weight'length);")

		# v_reset
		self.tb.architecture.processes["v_reset_gen"].bodyHeader.\
				add("v_reset <= to_signed(1000000, "
				"v_reset'length);")

		# v_th_value
		self.tb.architecture.processes["v_th_value_gen"].bodyHeader.\
				add("v_th_value <= to_signed(1500000000, "
				"v_th_value'length);")

		# rst_n
		self.tb.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '1';")
		self.tb.architecture.processes["rst_n_gen"].bodyHeader.add(
				"wait for 15 ns;")
		self.tb.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '0';")
		self.tb.architecture.processes["rst_n_gen"].bodyHeader.add(
				"wait for 10 ns;")
		self.tb.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '1';")

		# v_th_en
		neuron_load_ready_if = If()
		neuron_load_ready_if._if_.conditions.add(
			"neuron_load_ready = '1'")
		neuron_load_ready_if._if_.body.add("v_th_en <= '1';")
		neuron_load_ready_if._else_.body.add("v_th_en <= '0';")


		self.tb.architecture.processes["v_th_en_gen"].final_wait = False
		self.tb.architecture.processes["v_th_en_gen"].sensitivity_list.\
			add("clk")
		self.tb.architecture.processes["v_th_en_gen"].if_list.add()
		self.tb.architecture.processes["v_th_en_gen"].if_list[0]._if_.\
			conditions.add("clk'event")
		self.tb.architecture.processes["v_th_en_gen"].if_list[0]._if_.\
			conditions.add("clk = '1'", "and")
		self.tb.architecture.processes["v_th_en_gen"].if_list[0]._if_.\
			body.add(neuron_load_ready_if)

		# neuron_load_end
		neuron_load_ready_if = If()
		neuron_load_ready_if._if_.conditions.add(
			"neuron_load_ready = '1'")
		neuron_load_ready_if._if_.body.add("neuron_load_end <= '1';")
		neuron_load_ready_if._else_.body.add("neuron_load_end <= '0';")


		self.tb.architecture.processes["neuron_load_end_gen"].\
			final_wait = False
		self.tb.architecture.processes["neuron_load_end_gen"].\
			sensitivity_list.add("clk")
		self.tb.architecture.processes["neuron_load_end_gen"].if_list.\
			add()
		self.tb.architecture.processes["neuron_load_end_gen"].\
			if_list[0]._if_.conditions.add("clk'event")
		self.tb.architecture.processes["neuron_load_end_gen"].\
			if_list[0]._if_.conditions.add("clk = '1'", "and")
		self.tb.architecture.processes["neuron_load_end_gen"].\
				if_list[0]._if_.body.add(neuron_load_ready_if)

		# Start
		mc_ready_if = If()
		mc_ready_if._if_.conditions.add("mc_ready = '1'")
		mc_ready_if._if_.body.add("start <= '1';")
		mc_ready_if._else_.body.add("start <= '0';")


		self.tb.architecture.processes["start_gen"].final_wait = False
		self.tb.architecture.processes["start_gen"].sensitivity_list.\
			add("clk")
		self.tb.architecture.processes["start_gen"].if_list.add()
		self.tb.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk'event")
		self.tb.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk = '1'", "and")
		self.tb.architecture.processes["start_gen"].if_list[0]._if_.\
			body.add(mc_ready_if)


		del self.tb.architecture.processes["exc_spikes_gen"]
		del self.tb.architecture.processes["inh_spikes_gen"]
		self.tb.load(signal_name = "exc_spikes", input_dir = input_dir)
		self.tb.load(signal_name = "inh_spikes", input_dir = input_dir)

		del self.tb.architecture.processes["exc_spikes_rd_en_gen"]
		self.tb.architecture.bodyCodeHeader.add("exc_spikes_rd_en <= "
				"start_all;")
		del self.tb.architecture.processes["inh_spikes_rd_en_gen"]
		self.tb.architecture.bodyCodeHeader.add("inh_spikes_rd_en <= "
				"start_all;")



a = MultiCycleLIF(
	n_exc_inputs = 4,
	n_inh_inputs = 3,
	n_cycles = 1,
	bitwidth = 32,
	w_inh_bw = 32,
	w_exc_bw = 32,
	shift = 10,
	debug = True,
	debug_list = [
		"neuron_cu_present_state",
		"multi_input_cu_present_state",
		"multi_cycle_cu_present_state",
		"multi_cycle_datapath_cycles_cnt",
		"neuron_datapath_v",
		"multi_cycle_stop"
	]
)

a.testbench()

a.tb.write_file_all()
a.compile_all()
a.tb.compile()
a.tb.elaborate()
