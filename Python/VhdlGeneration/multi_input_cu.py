import subprocess as sp

import path_config

from vhdl_block import VHDLblock
from if_statement import If

from spiker_pkg import SpikerPackage
from utils import track_signals


class MultiInputCU(VHDLblock):

	def __init__(self, debug = False):

		VHDLblock.__init__(self, entity_name = "multi_input_cu")

		self.spiker_pkg = SpikerPackage()

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
				name 		= "load_end", 
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
				name 		= "exc_yes", 
				direction	= "in",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "exc_stop", 
				direction	= "in",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "inh_yes", 
				direction	= "in",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "inh_stop", 
				direction	= "in",
				port_type	= "std_logic")

		# Output towards datapath
		self.entity.port.add(
				name 		= "exc_cnt_en", 
				direction	= "out",
				port_type	= "std_logic")
		self.entity.port.add(
				name 		= "exc_cnt_rst_n", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "inh_cnt_en",
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "inh_cnt_rst_n", 
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
				name 		= "spike_sample", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "spike_rst_n", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "neuron_restart", 
				direction	= "out",
				port_type	= "std_logic")

		# Output towards outside
		self.entity.port.add(
				name 		= "load_ready", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "ready", 
				direction	= "out",
				port_type	= "std_logic")



		# Signals
		self.architecture.signal.add(
				name = "present_state",
				signal_type = "mi_states")

		self.architecture.signal.add(
				name = "next_state",
				signal_type = "mi_states")

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

		for state in self.spiker_pkg.pkg_dec.type_list["mi_states"].\
			typeElement:

			self.architecture.processes["state_evaluation"].\
					case_list["present_state"].when_list.\
					add(state)

		# Reset
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("next_state <= load;")

		# Load
		load_end_check = If()
		load_end_check._if_.conditions.add("load_end = '1'")
		load_end_check._if_.body.add("next_state <= idle_wait;")
		load_end_check._else_.body.add("next_state <= load;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["load"].\
				body.add(load_end_check)

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
		start_check._if_.body.add("next_state <= sample;")
		start_check._else_.body.add("next_state <= idle;")

		restart_check = If()
		restart_check._if_.conditions.add("restart = '1'")
		restart_check._if_.body.add("next_state <= init_wait;")
		restart_check._else_.body.add(start_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add(restart_check)

		# Init wait
		all_ready_check = If()
		all_ready_check._if_.conditions.add("all_ready = '1'")
		all_ready_check._if_.body.add("next_state <= init;")
		all_ready_check._else_.body.add("next_state <= init_wait;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"init_wait"].body.add(all_ready_check)
				
		# Sample 
		inh_check_0 = If()
		inh_check_0._if_.conditions.add("inh_yes = '1'")
		inh_check_0._if_.body.add("next_state <= exc_inh_wait;")
		inh_check_0._else_.body.add("next_state <= exc_wait;")

		inh_check_1 = If()
		inh_check_1._if_.conditions.add("inh_yes = '1'")
		inh_check_1._if_.body.add("next_state <= inh_wait;")
		inh_check_1._else_.body.add("next_state <= idle;")

		exc_check = If()
		exc_check._if_.conditions.add("exc_yes = '1'")
		exc_check._if_.body.add(inh_check_0)
		exc_check._else_.body.add(inh_check_1)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["sample"].\
				body.add(exc_check)

		# Exc inh wait
		all_ready_check = If()
		all_ready_check._if_.conditions.add("all_ready = '1'")
		all_ready_check._if_.body.add("next_state <= exc_update_full;")
		all_ready_check._else_.body.add("next_state <= exc_inh_wait;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"exc_inh_wait"].body.add(all_ready_check)

		# Exc update full
		exc_stop_check = If()
		exc_stop_check._if_.conditions.add("exc_stop = '1'")
		exc_stop_check._if_.body.add("next_state <= inh_wait_full;")
		exc_stop_check._else_.body.add("next_state <= exc_update_full;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"exc_update_full"].body.add(exc_stop_check)

		# Inh wait full
		all_ready_check = If()
		all_ready_check._if_.conditions.add("all_ready = '1'")
		all_ready_check._if_.body.add("next_state <= inh_update_full;")
		all_ready_check._else_.body.add("next_state <= inh_wait_full;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"inh_wait_full"].body.add(all_ready_check)

		# Inh update full
		inh_stop_check = If()
		inh_stop_check._if_.conditions.add("inh_stop = '1'")
		inh_stop_check._if_.body.add("next_state <= idle;")
		inh_stop_check._else_.body.add("next_state <= inh_update_full;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"inh_update_full"].body.add(inh_stop_check)

		# Exc wait
		all_ready_check = If()
		all_ready_check._if_.conditions.add("all_ready = '1'")
		all_ready_check._if_.body.add("next_state <= exc_update;")
		all_ready_check._else_.body.add("next_state <= exc_wait;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"exc_wait"].body.add(all_ready_check)

		# Exc update 
		exc_stop_check = If()
		exc_stop_check._if_.conditions.add("exc_stop = '1'")
		exc_stop_check._if_.body.add("next_state <= idle;")
		exc_stop_check._else_.body.add("next_state <= exc_update;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"exc_update"].body.add(exc_stop_check)

		# Inh wait
		all_ready_check = If()
		all_ready_check._if_.conditions.add("all_ready = '1'")
		all_ready_check._if_.body.add("next_state <= inh_update;")
		all_ready_check._else_.body.add("next_state <= inh_wait;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"inh_wait"].body.add(all_ready_check)

		# Inh update 
		inh_stop_check = If()
		inh_stop_check._if_.conditions.add("inh_stop = '1'")
		inh_stop_check._if_.body.add("next_state <= idle;")
		inh_stop_check._else_.body.add("next_state <= inh_update;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list[
				"inh_update"].body.add(inh_stop_check)



		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].others.\
				body.add("next_state <= reset;")

		# Output evaluation
		self.architecture.processes.add("output_evaluation")
		self.architecture.processes["output_evaluation"].\
				sensitivity_list.add("present_state")

		self.architecture.processes["output_evaluation"].\
				case_list.add("present_state")

		for state in self.spiker_pkg.pkg_dec.type_list["mi_states"].\
			typeElement:

			self.architecture.processes["output_evaluation"].\
					case_list["present_state"].when_list.\
					add(state)

		# Reset
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("exc_cnt_rst_n <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("inh_cnt_rst_n <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("spike_rst_n <= '0';")

		# Load
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["load"].\
				body.add("load_ready <= '1';")

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
				body.add("exc_cnt_rst_n <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add("inh_cnt_rst_n <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add("spike_rst_n <= '0';")

		# Init wait
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"init_wait"].body.add("ready <= '0';")
				
		# Sample 
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["sample"].\
				body.add("exc <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["sample"].\
				body.add("inh <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["sample"].\
				body.add("spike_sample <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["sample"].\
				body.add("ready <= '1';")

		# Exc inh wait
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"exc_inh_wait"].body.add("ready <= '0';")

		# Exc update full
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"exc_update_full"].body.add("exc <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"exc_update_full"].body.add(
				"exc_cnt_en <= '1';")

		# Inh wait full
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"inh_wait_full"].body.add("ready <= '0';")

		# Inh update full
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"inh_update_full"].body.add("inh <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"inh_update_full"].body.add(
				"inh_cnt_en <= '1';")

		# Exc wait
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"exc_wait"].body.add("ready <= '0';")

		# Exc update 
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"exc_update"].body.add("exc <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"exc_update"].body.add(
				"exc_cnt_en <= '1';")

		# Inh wait
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"inh_wait"].body.add("ready <= '0';")

		# Inh update 
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"inh_update"].body.add("inh <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list[
				"inh_update"].body.add(
				"inh_cnt_en <= '1';")



		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].others.\
				body.add("exc_cnt_rst_n <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].others.\
				body.add("inh_cnt_rst_n <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].others.\
				body.add("spike_rst_n <= '0';")


	def write_file_all(self, output_dir = "output"):
		self.spiker_pkg.write_file(output_dir = output_dir)
		self.write_file(output_dir = output_dir)
		

	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")

	def compile_all(self, output_dir = "output"):
		self.spiker_pkg.compile(output_dir = output_dir)
		self.compile(output_dir = output_dir)

	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")
