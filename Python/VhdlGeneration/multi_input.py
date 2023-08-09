import subprocess as sp
from math import log2

import path_config

from vhdl_block import VHDLblock
from multi_input_dp import MultiInputDP
from multi_input_cu import MultiInputCU
from testbench import Testbench

from spiker_pkg import SpikerPackage
from utils import track_signals, ceil_pow2


class MultiInput(VHDLblock):

	def __init__(self, n_exc_inputs = 2, n_inh_inputs = 2, debug = False):


		self.n_exc_inputs = n_exc_inputs
		self.n_inh_inputs = n_inh_inputs

		exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))

		VHDLblock.__init__(self, entity_name = "multi_input")

		self.spiker_pkg = SpikerPackage()

		self.datapath = MultiInputDP(
			n_exc_inputs = n_exc_inputs,
			n_inh_inputs = n_inh_inputs,
			debug = debug
		)

		self.control_unit = MultiInputCU(debug = debug)

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

		# INput controls
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
				name 		= "load_end", 
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
				name 		= "load_ready", 
				direction	= "out",
				port_type	= "std_logic")

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
			name 		= "exc_spike", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "exc_stop", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "inh_yes", 
			signal_type	= "std_logic")
		self.architecture.signal.add(
			name 		= "inh_spike", 
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


		if(debug):

			if self.datapath.debug:
				for debug_port in self.datapath.debug:

					debug_port_name = debug_port + "_out"

					self.entity.port.add(
						name 		=
							debug_port_name, 
						direction	= "out",
						port_type	= self.\
							datapath.entity.\
							port[debug_port_name].\
							port_type
					)

			if self.control_unit.debug:
				for debug_port in self.control_unit.debug:

					debug_port_name = debug_port + "_out"

					self.entity.port.add(
						name 		= 
							debug_port_name, 
						direction	= "out",
						port_type	= self.\
							control_unit.entity.\
							port[debug_port_name].\
							port_type
					)


			debug_signals = track_signals(self.architecture.signal,
					self.entity.name)

			for debug_port in debug_signals:

				debug_port_name = debug_port + "_out"

				self.entity.port.add(
					name 		= debug_port_name, 
					direction	= "out",
					port_type	= self.architecture.\
							signal[debug_port].\
							signal_type)

				# Bring the signal out
				connect_string = debug_port_name + " <= " + \
							debug_port + ";"
				self.architecture.bodyCodeHeader.\
						add(connect_string)

	
		

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
