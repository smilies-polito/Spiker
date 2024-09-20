from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If


class MultiCycleCU(VHDLblock):

	def __init__(self, debug = False, debug_list = []):

		self.name = "multi_cycle_cu"

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
				name 		= "start", 
				direction	= "in",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "all_ready", 
				direction	= "in",
				port_type	= "std_logic")

		# Input from datapath
		self.entity.port.add(
				name 		= "stop", 
				direction	= "in",
				port_type	= "std_logic")

		# Output towards datapath
		self.entity.port.add(
				name 		= "cycles_cnt_en",
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "cycles_cnt_rst_n", 
				direction	= "out",
				port_type	= "std_logic")

		# Output towards outside
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
				name = "present_state",
				signal_type = "mc_states")

		self.architecture.signal.add(
				name = "next_state",
				signal_type = "mc_states")

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



		self.architecture.processes.add("state_evaluation")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("present_state")

		for key in self.entity.port:
			if self.entity.port[key].direction == "in" and \
				key != "clk" and key != "rst_n":

				self.architecture.processes\
					["state_evaluation"].\
					sensitivity_list.add(key)

		self.architecture.processes["state_evaluation"].\
				case_list.add("present_state")

		for state in self.spiker_pkg.pkg_dec.type_list["mc_states"].\
			typeElement:

			self.architecture.processes["state_evaluation"].\
					case_list["present_state"].when_list.\
					add(state)

		# Reset
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("next_state <= idle_wait;")

		# Idle wait
		all_ready_check = If()
		all_ready_check._if_.conditions.add("all_ready = '1'")
		all_ready_check._if_.body.add("next_state <= idle;")
		all_ready_check._else_.body.add("next_state <= idle_wait;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["idle_wait"].\
				body.add(all_ready_check)

		# Idle
		start_check = If()
		start_check._if_.conditions.add("start = '1'")
		start_check._if_.body.add("next_state <= init;")
		start_check._else_.body.add("next_state <= idle;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add(start_check)

		# Init
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["init"].\
				body.add("next_state <= update_wait;")
				
		# Update wait
		stop_check = If()
		stop_check._if_.conditions.add("stop = '1'")
		stop_check._if_.body.add("next_state <= idle;")
		stop_check._else_.body.add("next_state <= network_update;")

		all_ready_check = If()
		all_ready_check._if_.conditions.add("all_ready = '1'")
		all_ready_check._if_.body.add(stop_check)
		all_ready_check._else_.body.add("next_state <= update_wait;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["update_wait"].\
				body.add(all_ready_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].others.body.add(
				"next_state <= reset;")

		# Network update
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"network_update"].body.add("next_state <= "
				"update_wait;")



		self.architecture.processes.add("output_evaluation")
		self.architecture.processes["output_evaluation"].\
				sensitivity_list.add("present_state")

		# Default values
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("ready <= '0';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("start_all <= '0';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("cycles_cnt_en <= '0';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("cycles_cnt_rst_n <= '1';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("restart <= '0';")

		self.architecture.processes["output_evaluation"].\
				case_list.add("present_state")

		for state in self.spiker_pkg.pkg_dec.type_list["mc_states"].\
			typeElement:

			self.architecture.processes["output_evaluation"].\
					case_list["present_state"].when_list.\
					add(state)

		# Reset
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("cycles_cnt_rst_n <= '0';")

		# Idle wait
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"idle_wait"].body.add("ready <= '0';")

		# Idle
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add("ready <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add("cycles_cnt_rst_n <= '0';")

		# Init
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["init"].\
				body.add("restart <= '1';")
				
		# Update wait 
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["update_wait"].\
				body.add("start_all <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["update_wait"].\
				body.add("cycles_cnt_en <= '0';")

		# Network update
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["network_update"].\
				body.add("start_all <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["network_update"].\
				body.add("cycles_cnt_en <= '1';")


		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].others.body.add(
				"cycles_cnt_rst_n <= '0';")

				

		# Debug
		if debug:
			debug_component(self, debug_list)


	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
