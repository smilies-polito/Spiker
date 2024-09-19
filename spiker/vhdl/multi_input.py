from math import log2

from .multi_input_dp import MultiInputDP
from .multi_input_cu import MultiInputCU
from .testbench import Testbench
from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2

from .vhdltools.vhdl_block import VHDLblock

class MultiInput(VHDLblock):

	def __init__(self, n_exc_inputs = 2, n_inh_inputs = 2, debug = False,
			debug_list = []):

		self.name = "multi_input_" + str(n_exc_inputs) + "_exc_" + \
			str(n_inh_inputs) + "_inh"

		self.n_exc_inputs = n_exc_inputs
		self.n_inh_inputs = n_inh_inputs
		self.exc_cnt_bitwidth = int(log2(ceil_pow2(n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(n_inh_inputs)))

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

		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)

	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

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
			name		= "exc_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.exc_cnt_bitwidth))
		self.entity.generic.add(
			name		= "inh_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.inh_cnt_bitwidth))

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
			name 		= "neurons_ready", 
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
			name 		= "out_sample", 
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

		self.architecture.bodyCodeHeader.add("out_sample <= "
				"spike_sample;")
		
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
	
		
	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
