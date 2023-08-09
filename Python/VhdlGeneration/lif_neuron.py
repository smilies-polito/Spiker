import subprocess as sp

import path_config
from vhdl_block import VHDLblock
from lif_neuron_dp import LIFneuronDP
from lif_neuron_cu import LIFneuronCU
from and_mask import AndMask
from testbench import Testbench

from spiker_pkg import SpikerPackage
from utils import track_signals, debug_component


class LIFneuron(VHDLblock):

	def __init__(self, bitwidth = 16, w_inh_bw =
			5, w_exc_bw = 5,
			shift = 10, debug = False):

		VHDLblock.__init__(self, entity_name = "neuron")

		self.spiker_pkg = SpikerPackage()

		self.datapath = LIFneuronDP(
			bitwidth = bitwidth,
			w_inh_bw = 
			w_inh_bw,
			w_exc_bw = 
			w_exc_bw, 
			shift = shift,
			debug = debug
		)

		self.control_unit = LIFneuronCU(debug = debug)
		self.and_mask = AndMask(data_type = "signed")

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
			name 		= "restart", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "load_end", 
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
			name 		= "load_ready",
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
		self.architecture.signal.add(
			name 		= "v_update",
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "masked_v_th_en",
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
		self.architecture.signal.add(
			name 		= "load_ready_fb",
			signal_type	= "std_logic")


		if w_inh_bw < bitwidth:
			self.architecture.signal.add(
				name 		= "masked_inh_weight",
				signal_type	= "signed("
						"inh_weights_bit_width-1 "
						"downto 0)")
		elif w_inh_bw == bitwidth:
			self.architecture.signal.add(
				name 		= "masked_inh_weight",
				signal_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			print("Inhibitory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)
			

		if w_exc_bw < bitwidth:
			self.architecture.signal.add(
				name 		= "masked_exc_weight",
				signal_type	= "signed("
					"exc_weights_bit_width-1 downto 0)")

		elif w_exc_bw == bitwidth:
			self.architecture.signal.add(
				name 		= "masked_exc_weight",
				signal_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			print("Excitatory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)




		# Components
		self.architecture.component.add(self.datapath)
		self.architecture.component.add(self.control_unit)
		self.architecture.component.add(self.and_mask)

		# Mask threshold enable
		self.architecture.bodyCodeHeader.add("masked_v_th_en <= "
				"v_th_en and load_ready_fb;")
		self.architecture.bodyCodeHeader.add("load_ready <= "
				"load_ready_fb;")

		
		# Datapath
		self.architecture.instances.add(self.datapath,
				"datapath")
		self.architecture.instances["datapath"].generic_map()
		self.architecture.instances["datapath"].port_map()
		self.architecture.instances["datapath"].p_map.add("v_th_en",
				"masked_v_th_en")
		self.architecture.instances["datapath"].p_map.add("exc_weight",
				"masked_exc_weight")
		self.architecture.instances["datapath"].p_map.add("inh_weight",
				"masked_inh_weight")

		# Control unit
		self.architecture.instances.add(self.control_unit,
				"control_unit")
		self.architecture.instances["control_unit"].generic_map()
		self.architecture.instances["control_unit"].port_map()
		self.architecture.instances["control_unit"].p_map.add(
				"load_ready", "load_ready_fb")

		# Excitatory weights mask
		self.architecture.instances.add(self.and_mask,
				"exc_mask")
		self.architecture.instances["exc_mask"].generic_map()
		self.architecture.instances["exc_mask"].port_map()

		if w_inh_bw < bitwidth:
			self.architecture.instances["exc_mask"].g_map.add("N",
				"exc_weights_bit_width")

		elif w_inh_bw == bitwidth:
			self.architecture.instances["exc_mask"].g_map.add("N",
				"neuron_bit_width")
		else:
			print("Excitatory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)

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

		if w_inh_bw < bitwidth:
			self.architecture.instances["inh_mask"].g_map.add("N",
				"inh_weights_bit_width")

		elif w_inh_bw == bitwidth:
			self.architecture.instances["inh_mask"].g_map.add("N",
				"neuron_bit_width")
		else:
			print("Inhibitory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)

		self.architecture.instances["inh_mask"].p_map.add("input_bits",
				"inh_weight")
		self.architecture.instances["inh_mask"].p_map.add("mask_bit",
				"inh_spike")
		self.architecture.instances["inh_mask"].p_map.add("output_bits",
				"masked_inh_weight")


		# Debug
		if debug:
			debug_component(self)
	
		

	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")


	def compile_all(self, output_dir = "output"):

		self.spiker_pkg.compile(output_dir = output_dir)
		self.datapath.compile_all(output_dir = output_dir)
		self.control_unit.compile(output_dir = output_dir)
		self.and_mask.compile(output_dir = output_dir)

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")


	def write_file_all(self, output_dir = "output"):

		self.spiker_pkg.write_file(output_dir = output_dir)
		self.datapath.write_file_all(output_dir = output_dir)
		self.control_unit.write_file(output_dir = output_dir)
		self.and_mask.write_file(output_dir = output_dir)
		self.write_file(output_dir = output_dir)



	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")


	def testbench(self, clock_period = 20, file_output = False):

		self.tb = Testbench(self, clock_period = clock_period,
			file_output = file_output)

		self.tb.library.add("work")
		self.tb.library["work"].package.add("spiker_pkg")

		# exc_weight
		self.tb.architecture.processes["exc_weight_gen"].bodyHeader.\
				add("exc_weight <= to_signed(500, "
				"exc_weight'length);")

		# inh_weight
		self.tb.architecture.processes["inh_weight_gen"].bodyHeader.\
				add("inh_weight <= to_signed(-300, "
				"inh_weight'length);")

		# v_reset
		self.tb.architecture.processes["v_reset_gen"].bodyHeader.\
				add("v_reset <= to_signed(1000, "
				"v_reset'length);")

		# v_th_value
		self.tb.architecture.processes["v_th_value_gen"].bodyHeader.\
				add("v_th_value <= to_signed(3000, "
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
		self.tb.architecture.processes["v_th_en_gen"].bodyHeader.add(
				"v_th_en <= '0';")
		self.tb.architecture.processes["v_th_en_gen"].bodyHeader.add(
				"wait for 50 ns;")
		self.tb.architecture.processes["v_th_en_gen"].bodyHeader.add(
				"v_th_en <= '1';")
		self.tb.architecture.processes["v_th_en_gen"].bodyHeader.add(
				"wait for 20 ns;")
		self.tb.architecture.processes["v_th_en_gen"].bodyHeader.add(
				"v_th_en <= '0';")

		# load_end
		self.tb.architecture.processes["load_end_gen"].bodyHeader.add(
				"load_end <= '0';")
		self.tb.architecture.processes["load_end_gen"].bodyHeader.add(
				"wait for 50 ns;")
		self.tb.architecture.processes["load_end_gen"].bodyHeader.add(
				"load_end <= '1';")
		self.tb.architecture.processes["load_end_gen"].bodyHeader.add(
				"wait for 20 ns;")
		self.tb.architecture.processes["load_end_gen"].bodyHeader.add(
				"load_end <= '0';")

		# restart
		self.tb.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '0';")
		self.tb.architecture.processes["restart_gen"].bodyHeader.add(
				"wait for 70 ns;")
		self.tb.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '1';")
		self.tb.architecture.processes["restart_gen"].bodyHeader.add(
				"wait for 20 ns;")
		self.tb.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '0';")

		# exc
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '0';")
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"wait for 130 ns;")
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '1';")
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"wait for 600 ns;")
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '0';")
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"wait for 100 ns;")
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '1';")
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"wait for 60 ns;")
		self.tb.architecture.processes["exc_gen"].bodyHeader.add(
				"exc <= '0';")

		# exc spike
		self.tb.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"exc_spike <= '0';")
		self.tb.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"wait for 130 ns;")
		self.tb.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"exc_spike <= '1';")
		self.tb.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"wait for 600 ns;")
		self.tb.architecture.processes["exc_spike_gen"].bodyHeader.add(
				"exc_spike <= '0';")

		# inh
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '0';")
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"wait for 750 ns;")
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '1';")
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"wait for 60 ns;")
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '0';")
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"wait for 20 ns;")
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '1';")
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"wait for 60 ns;")
		self.tb.architecture.processes["inh_gen"].bodyHeader.add(
				"inh <= '0';")

		# inh spike
		self.tb.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"inh_spike <= '0';")
		self.tb.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"wait for 750 ns;")
		self.tb.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"inh_spike <= '1';")
		self.tb.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"wait for 60 ns;")
		self.tb.architecture.processes["inh_spike_gen"].bodyHeader.add(
				"inh_spike <= '0';")
