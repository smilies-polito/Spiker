from math import log2

from multi_cycle_dp import MultiCycleDP
from multi_cycle_cu import MultiCycleCU
from testbench import Testbench
from spiker_pkg import SpikerPackage
from utils import track_signals, ceil_pow2, debug_component

import path_config
from vhdl_block import VHDLblock

class MultiCycle(VHDLblock):

	def __init__(self, n_cycles = 2, debug = False, debug_list = []):

		self.n_cycles = n_cycles

		cycles_cnt_bitwidth = int(log2(ceil_pow2(n_cycles+1))) + 1

		VHDLblock.__init__(self, entity_name = "multi_cycle")

		self.spiker_pkg = SpikerPackage()

		self.datapath = MultiCycleDP(
			n_cycles = n_cycles,
			debug = debug,
			debug_list = debug_list
		)

		self.control_unit = MultiCycleCU(
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
			name		= "cycles_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(cycles_cnt_bitwidth))
		self.entity.generic.add(
			name		= "n_cycles", 
			gen_type	= "integer",
			value		= str(n_cycles))

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

		self.entity.port.add(
				name 		= "all_ready", 
				direction	= "in",
				port_type	= "std_logic")

		# Output
		self.entity.port.add(
			name 		= "ready", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "restart", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "start_all", 
			direction	= "out",
			port_type	= "std_logic")

		# Signals
		self.architecture.signal.add(
			name 		= "cycles_cnt_en", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "cycles_cnt_rst_n", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "stop", 
			signal_type	= "std_logic")


		# Components
		self.architecture.component.add(self.datapath)
		self.architecture.component.add(self.control_unit)
		
		# Datapath
		self.architecture.instances.add(self.datapath,
				"datapath")
		self.architecture.instances["datapath"].generic_map()
		self.architecture.instances["datapath"].port_map()

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
