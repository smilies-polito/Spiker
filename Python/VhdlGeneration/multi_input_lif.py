import subprocess as sp
from math import log2

import path_config

from vhdl_block import VHDLblock
from multi_input import MultiInput
from lif_neuron import LIFneuron
from testbench import Testbench

from spiker_pkg import SpikerPackage
from utils import track_signals, ceil_pow2


class MultiInputLIF(VHDLblock):

	def __init__(self, n_exc_inputs = 2, n_inh_inputs = 2, bitwidth = 16,
			w_inh_bw = 5, w_exc_bw = 5, shift = 10,debug = False):

		self.n_exc_inputs = n_exc_inputs
		self.n_inh_inputs = n_inh_inputs

		exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))

		VHDLblock.__init__(self, entity_name = "multi_input_lif")

		self.spiker_pkg = SpikerPackage()

		self.multi_input = MultiInput(
			n_exc_inputs = n_exc_inputs,
			n_inh_inputs = n_inh_inputs,
			debug = debug
		)

		self.lif_neuron = LIFneuron(
			bitwidth = bitwidth,
			w_inh_bw = w_inh_bw,
			w_exc_bw = w_exc_bw,
			shift = shift,
			debug = debug
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
			name		= "exc_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(exc_cnt_bitwidth))
		self.entity.generic.add(
			name		= "inh_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(inh_cnt_bitwidth))
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
			name 		= "mi_ready", 
			direction	= "out",
			port_type	= "std_logic")

		# Output
		self.entity.port.add(
			name 		= "neuron_load_ready",
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
		

		# Components
		self.architecture.component.add(self.multi_input)
		self.architecture.component.add(self.lif_neuron)
		
		# Multi-input
		self.architecture.instances.add(self.multi_input,
				"mi")
		self.architecture.instances["mi"].generic_map()
		self.architecture.instances["mi"].port_map()

		# LIF neuron
		self.architecture.instances.add(self.lif_neuron,
				"lif_neuron")
		self.architecture.instances["lif_neuron"].generic_map()
		self.architecture.instances["lif_neuron"].port_map()


		if(debug):

			if self.multi_input.debug:
				for debug_port in self.multi_input.debug:

					debug_port_name = debug_port + "_out"

					self.entity.port.add(
						name 		=
							debug_port_name, 
						direction	= "out",
						port_type	= self.\
							multi_input.entity.\
							port[debug_port_name].\
							port_type
					)

			if self.lif_neuron.debug:
				for debug_port in self.lif_neuron.debug:

					debug_port_name = debug_port + "_out"

					self.entity.port.add(
						name 		= 
							debug_port_name, 
						direction	= "out",
						port_type	= self.\
							lif_neuron.entity.\
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
		self.multi_input.compile_all(output_dir = output_dir)
		self.lif_neuron.compile_all(output_dir = output_dir)

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")


	def write_file_all(self, output_dir = "output"):

		self.spiker_pkg.write_file(output_dir = output_dir)
		self.multi_input.write_file_all(output_dir = output_dir)
		self.lif_neuron.write_file_all(output_dir = output_dir)
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

a = MultiInputLIF()
print(a.code())

a.write_file_all()
a.compile_all()
a.elaborate()
