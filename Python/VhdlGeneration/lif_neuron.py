import subprocess as sp

import path_config
from vhdl_block import VHDLblock
from lif_neuron_dp import LIFneuronDP
from lif_neuron_cu import LIFneuronCU

from utils import track_signals


class LIFneuron(VHDLblock):

	def __init__(self, default_bitwidth = 16, default_inh_weights_bitwidth =
			5, default_exc_weights_bitwidth = 5,
			default_shift = 10, debug = False):

		VHDLblock.__init__(self, entity_name = "neuron")

		self.datapath = LIFneuronDP(
			default_bitwidth = default_bitwidth,
			default_inh_weights_bitwidth = 
			default_inh_weights_bitwidth,
			default_exc_weights_bitwidth = 
			default_exc_weights_bitwidth, 
			default_shift = default_shift,
			debug = debug
		)

		self.control_unit = LIFneuronCU(debug = debug)

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
			value		= str(default_bitwidth))


		if default_inh_weights_bitwidth < default_bitwidth:
			self.entity.generic.add(
				name		= "inh_weights_bit_width",
				gen_type	= "integer",
				value		= str(
					default_inh_weights_bitwidth))

		if default_exc_weights_bitwidth < default_bitwidth:
			self.entity.generic.add(
				name		= "exc_weights_bit_width",
				gen_type	= "integer",
				value		= str(
					default_exc_weights_bitwidth))

		self.entity.generic.add(
			name		= "shift",
			gen_type	= "integer",
			value		= str(default_shift))

		# Input parameters
		self.entity.port.add(
			name 		= "v_th_value", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")
		self.entity.port.add(
			name 		= "v_reset", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")


		if default_inh_weights_bitwidth < default_bitwidth:
			self.entity.port.add(
				name 		= "inh_weight",
				direction	= "in",
				port_type	= "signed("
						"inh_weights_bit_width-1 "
						"downto 0)")
		elif default_inh_weights_bitwidth == default_bitwidth:
			self.entity.port.add(
				name 		= "inh_weight",
				direction	= "in",
				port_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			print("Inhibitory weight bit-width cannot be larger "
				"than the neuron's one")
			exit(-1)
			

		if default_exc_weights_bitwidth < default_bitwidth:
			self.entity.port.add(
				name 		= "exc_weight",
				direction	= "in",
				port_type	= "signed("
					"exc_weights_bit_width-1 downto 0)")

		elif default_exc_weights_bitwidth == default_bitwidth:
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
			name 		= "v_th_en", 
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

		self.datapath.compile_all()
		self.control_unit.compile()

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")


	def write_file_all(self, output_dir = "output"):

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
