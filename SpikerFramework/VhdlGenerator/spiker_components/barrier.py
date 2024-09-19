from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .barrier_cu import BarrierCU
from .reg import Reg

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If

class Barrier(VHDLblock):

	def __init__(self, bitwidth = 1, debug = False, debug_list = []):

		self.name = "barrier"

		self.bitwidth = bitwidth

		self.spiker_pkg = SpikerPackage()

		self.cu		= BarrierCU()

		self.reg	= Reg(
			bitwidth	= bitwidth,
			reg_type	= "std_logic_vector",
			rst		= "sync",
			active		= "low",
			debug 		= debug,
			debug_list	= debug_list
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
				name 		= "N", 
				gen_type	= "integer",
				value		= str(self.bitwidth)
		)


		# Input from outside
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
				name 		= "out_sample", 
				direction	= "in",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "reg_in", 
				direction	= "in",
				port_type	= "std_logic_vector(N-1"
				" downto 0)")


		# Output towards outside
		self.entity.port.add(
				name 		= "ready", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "reg_out", 
				direction	= "out",
				port_type	= "std_logic_vector(N-1"
				" downto 0)")


		self.architecture.signal.add(
				name = "present_state",
				signal_type = "mi_states")

		self.architecture.signal.add(
				name = "next_state",
				signal_type = "mi_states")

		# from CU to datapath
		self.architecture.signal.add(
				name 		= "barrier_rst_n", 
				signal_type	= "std_logic")

		self.architecture.signal.add(
				name 		= "barrier_en", 
				signal_type	= "std_logic")


		self.architecture.component.add(self.cu)
		self.architecture.component.add(self.reg)

		self.architecture.instances.add(self.cu, "control_unit")
		self.architecture.instances["control_unit"].generic_map()
		self.architecture.instances["control_unit"].port_map()

		self.architecture.instances.add(self.reg, "datapath")
		self.architecture.instances["datapath"].generic_map()
		self.architecture.instances["datapath"].port_map()
		self.architecture.instances["datapath"].p_map.add("en",
				"barrier_en")
		self.architecture.instances["datapath"].p_map.add("rst_n",
				"barrier_rst_n")


		# Debug
		if debug:
			debug_component(self, debug_list)


	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
