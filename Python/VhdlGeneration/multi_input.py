from math import log2

from multi_input_dp import MultiInputDP
from multi_input_cu import MultiInputCU
from testbench import Testbench
from spiker_pkg import SpikerPackage
from utils import track_signals, ceil_pow2, debug_component

import path_config
from vhdl_block import VHDLblock

class MultiInput(VHDLblock):

	def __init__(self, n_exc_inputs = 2, n_inh_inputs = 2, debug = False,
			debug_list = []):


		self.n_exc_inputs = n_exc_inputs
		self.n_inh_inputs = n_inh_inputs

		exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))

		VHDLblock.__init__(self, entity_name = "multi_input")

		self.spiker_pkg = SpikerPackage()

		self.datapath = MultiInputDP(
			n_exc_inputs = n_exc_inputs,
			n_inh_inputs = n_inh_inputs,
			debug = debug,
			debug_list = debug_list
		)

		self.control_unit = MultiInputCU(
			debug = debug,
			debug_list = debug_list
		)

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

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
			name		= "exc_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(exc_cnt_bitwidth))
		self.entity.generic.add(
			name		= "inh_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(inh_cnt_bitwidth))

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
		self.entity.port.add(
			name 		= "all_ready", 
			direction	= "in",
			port_type	= "std_logic")

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
			name 		= "ready", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "neuron_restart", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "exc", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "inh", 
			direction	= "out",
			port_type	= "std_logic")
		
		self.entity.port.add(
			name 		= "exc_spike", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "inh_spike", 
			direction	= "out",
			port_type	= "std_logic")

		# Signals
		self.architecture.signal.add(
			name 		= "spike_sample", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "spike_rst_n", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "exc_cnt_en", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "exc_cnt_rst_n", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "inh_cnt_en", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "inh_cnt_rst_n", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "exc_yes", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "exc_stop", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "inh_yes", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "inh_stop", 
			signal_type	= "std_logic")


		# Components
		self.architecture.component.add(self.datapath)
		self.architecture.component.add(self.control_unit)
		
		# Datapath
		self.architecture.instances.add(self.datapath,
				"datapath")
		self.architecture.instances["datapath"].generic_map()
		self.architecture.instances["datapath"].port_map()
		self.architecture.instances["datapath"].p_map.add("exc_sample",
				"spike_sample")
		self.architecture.instances["datapath"].p_map.add("inh_sample",
				"spike_sample")
		self.architecture.instances["datapath"].p_map.add("exc_rst_n",
				"spike_rst_n")
		self.architecture.instances["datapath"].p_map.add("inh_rst_n",
				"spike_rst_n")

		# Control unit
		self.architecture.instances.add(self.control_unit,
				"control_unit")
		self.architecture.instances["control_unit"].generic_map()
		self.architecture.instances["control_unit"].port_map()


		# Debug
		if debug:
			debug_component(self, debug_list)
	
		
	def compile_all(self, output_dir = "output"):

		self.spiker_pkg.compile(output_dir = output_dir)
		self.datapath.compile_all(output_dir = output_dir)
		self.control_unit.compile(output_dir = output_dir)

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
		self.write_file(output_dir = output_dir)
