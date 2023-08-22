import subprocess as sp
import numpy as np

from math import log2

from multi_cycle_lif import MultiCycleLIF
from rom import Rom
from testbench import Testbench
from spiker_pkg import SpikerPackage
from utils import track_signals, ceil_pow2, debug_component

import path_config
from vhdl_block import VHDLblock
from if_statement import If



class SingleLifBram(VHDLblock):

	def __init__(self,  n_cycles = 10,
			bitwidth = 16, w_inh_bw = 5, w_exc_bw = 5, shift = 10,
			w_exc_array = np.array([[0.1, 0.2]]), w_inh_array =
			np.array([[-0.1, -0.2]]), debug = False, 
			debug_list = []):

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

		VHDLblock.__init__(self, entity_name = "single_lif_bram")

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


		self.vhdl(debug = debug)



	def vhdl(self, debug = False):
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
			name 		= "v_th_value", 
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
		

		# Components
		self.architecture.component.add(self.lif_neuron)
		self.architecture.component.add(self.exc_mem)
		self.architecture.component.add(self.inh_mem)

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
				"addra", "exc_cnt")
		self.architecture.instances["exc_mem"].p_map.add(
				"clka", "clk")
	

		# Inhibitory memory
		self.architecture.instances.add(self.inh_mem,
				"inh_mem")
		self.architecture.instances["inh_mem"].generic_map()
		self.architecture.instances["inh_mem"].port_map()
		self.architecture.instances["inh_mem"].p_map.add(
				"dout_0", "inh_weight")
		self.architecture.instances["inh_mem"].p_map.add(
				"addra", "inh_cnt")
		self.architecture.instances["inh_mem"].p_map.add(
				"clka", "clk")


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

		attr_list = [ attr for attr in dir(self) if not 
				attr.startswith("__")]

		for attr_name in attr_list:

			sub_component = getattr(self, attr_name)

			if hasattr(sub_component, "compile") and \
			callable(sub_component.compile):

				sub_component.compile(output_dir = output_dir)

		self.compile(output_dir = output_dir)


	def write_file_all(self, output_dir = "output"):

		attr_list = [ attr for attr in dir(self) if not 
				attr.startswith("__")]

		for attr_name in attr_list:

			sub_component = getattr(self, attr_name)

			if hasattr(sub_component, "write_file") and \
			callable(sub_component.write_file):

				sub_component.write_file(
					output_dir = output_dir)

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



# a = SingleLifBram(
# 	n_cycles = 1,
# 	bitwidth = 32,
# 	w_inh_bw = 32,
# 	w_exc_bw = 32,
# 	shift = 10,
# 	w_exc_array = np.array([[0.1, 0.2, 0.3, 0.4]]),
# 	w_inh_array = np.array([[-0.1, -0.2, -0.3, -0.4]]),
# 	debug = False,
# 	debug_list = [
# 		"neuron_cu_present_state",
# 		"multi_input_cu_present_state",
# 		"multi_cycle_cu_present_state",
# 		"multi_cycle_datapath_cycles_cnt",
# 		"neuron_datapath_v",
# 		"multi_cycle_stop"
# 	]
# )


a = SingleLifBram()

a.testbench()

a.tb.write_file_all()
a.tb.compile_all()
a.tb.elaborate()
