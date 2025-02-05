from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If

class BarrierCU(VHDLblock):

	def __init__(self, debug = False, debug_list = []):

		self.name = "barrier_cu"

		self.spiker_pkg = SpikerPackage()
		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		self.library.add("work")
		self.library["work"].package.add("spiker_pkg")


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

		# Output towards datapath
		self.entity.port.add(
				name 		= "barrier_rst_n", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "barrier_en", 
				direction	= "out",
				port_type	= "std_logic")


		# Output towards outside
		self.entity.port.add(
				name 		= "ready", 
				direction	= "out",
				port_type	= "std_logic")


		self.architecture.signal.add(
				name = "present_state",
				signal_type = "mi_states")

		self.architecture.signal.add(
				name = "next_state",
				signal_type = "mi_states")

		# State transition proces
		self.architecture.processes.add("state_transition")
		self.architecture.processes["state_transition"].\
				sensitivity_list.add("clk")
		self.architecture.processes["state_transition"].\
				sensitivity_list.add("rst_n")

		self.architecture.processes["state_transition"].if_list.add()
		self.architecture.processes["state_transition"].\
				if_list[0]._if_.conditions.add("rst_n = '0'")
		self.architecture.processes["state_transition"].\
				if_list[0]._if_.body.add(
				"present_state <= reset;")
		self.architecture.processes["state_transition"].\
				if_list[0]._elsif_.add()
		self.architecture.processes["state_transition"].\
				if_list[0]._elsif_[0].conditions.add(
				"clk'event")
		self.architecture.processes["state_transition"].\
				if_list[0]._elsif_[0].conditions.add(
						"clk = '1'", "and")
		self.architecture.processes["state_transition"].\
				if_list[0]._elsif_[0].body.add(
				"present_state <= next_state;")


		# State evaluation process
		self.architecture.processes.add("state_evaluation")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("present_state")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("restart")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("out_sample")

		self.architecture.processes["state_evaluation"].\
				case_list.add("present_state")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("reset")
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("next_state <= idle;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("idle")

		out_sample_check = If()
		out_sample_check._if_.conditions.add("out_sample = '1'")
		out_sample_check._if_.body.add("next_state <= sample;")
		out_sample_check._else_.body.add("next_state <= idle;")

		restart_check = If()
		restart_check._if_.conditions.add("restart = '1'")
		restart_check._if_.body.add("next_state <= init;")
		restart_check._else_.body.add(out_sample_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add(restart_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("init")
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["init"].\
				body.add("next_state <= idle;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("sample")
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["sample"].\
				body.add("next_state <= idle;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].\
				others.body.add("next_state <= idle;")

		
		# Output evaluation process
		self.architecture.processes.add("output_evaluation")
		self.architecture.processes["output_evaluation"].\
				sensitivity_list.add("present_state")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("ready <= '0';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("barrier_rst_n <= '1';")

		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("barrier_en <= '0';")


		self.architecture.processes["output_evaluation"].\
				case_list.add("present_state")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("reset")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("barrier_rst_n <= '0';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("idle")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add("ready <= '1';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("init")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["init"].\
				body.add("barrier_rst_n <= '0';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("sample")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["sample"].\
				body.add("barrier_en <= '1';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].\
				others.body.add("ready <= '1';")


		# Debug
		if debug:
			debug_component(self, debug_list)


	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
