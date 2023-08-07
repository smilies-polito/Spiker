import subprocess as sp

import path_config

from vhdl_block import VHDLblock
from if_statement import If
from utils import track_signals


class LIFneuronCU(VHDLblock):

	def __init__(self, debug = False):

		VHDLblock.__init__(self, entity_name = "neuron_cu")

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
				name 		= "exc", 
				direction	= "in",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "inh", 
				direction	= "in",
				port_type	= "std_logic")

		# Input from datapath
		self.entity.port.add(
				name 		= "exceed_v_th", 
				direction	= "in",
				port_type	= "std_logic")

		# Output towards datapath
		self.entity.port.add(
				name 		= "update_sel", 
				direction	= "out",
				port_type	= "std_logic_vector(1 downto"
							" 0)")
		self.entity.port.add(
				name 		= "add_or_sub", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "v_update", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "v_en", 
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
				name 		= "v_rst_n", 
				direction	= "out",
				port_type	= "std_logic")

		# Output towards outside
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

		self.architecture.signal.add(
				name = "present_state",
				signal_type = "neuron_states")

		self.architecture.signal.add(
				name = "next_state",
				signal_type = "neuron_states")

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
				if_list[0]._elsif_[0].conditions.add("clk'event")
		self.architecture.processes["state_transition"].\
				if_list[0]._elsif_[0].conditions.add("clk = '1'",
						"and")
		self.architecture.processes["state_transition"].\
				if_list[0]._elsif_[0].body.add(
				"present_state <= next_state;")



		self.architecture.processes.add("state_evaluation")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("present_state")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("load_end")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("restart")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("exc")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("inh")
		self.architecture.processes["state_evaluation"].\
				sensitivity_list.add("exceed_v_th")

		self.architecture.processes["state_evaluation"].\
				case_list.add("present_state")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("reset")
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("next_state <= load;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("load")

		load_end_check = If()
		load_end_check._if_.conditions.add("load_end = '1'")
		load_end_check._if_.body.add("next_state <= idle;")
		load_end_check._else_.body.add("next_state <= load;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["load"].\
				body.add(load_end_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("idle")

		inh_check_0 = If()
		inh_check_0._if_.conditions.add("inh = '1'")
		inh_check_0._if_.body.add("next_state <= inhibit;")
		inh_check_0._else_.body.add("next_state <= idle;")

		exceed_v_th_check = If()
		exceed_v_th_check._if_.conditions.add("exceed_v_th = '1'")
		exceed_v_th_check._if_.body.add("next_state <= fire;")
		exceed_v_th_check._else_.body.add("next_state <= leak;")

		inh_check_1 = If()
		inh_check_1._if_.conditions.add("inh = '1'")
		inh_check_1._if_.body.add(exceed_v_th_check)
		inh_check_1._else_.body.add("next_state <= excite;")

		exc_check = If()
		exc_check._if_.conditions.add("exc = '1'")
		exc_check._if_.body.add(inh_check_1)
		exc_check._else_.body.add(inh_check_0)

		restart_check = If()
		restart_check._if_.conditions.add("restart = '1'")
		restart_check._if_.body.add("next_state <= init;")
		restart_check._else_.body.add(exc_check)

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
				add("fire")

		exc_inh_check = If()
		exc_inh_check._if_.conditions.add("exc = '1'")
		exc_inh_check._if_.conditions.add("inh = '1'", "and")
		exc_inh_check._if_.body.add("next_state <= leak;")
		exc_inh_check._else_.body.add("next_state <= idle;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["fire"].\
				body.add(exc_inh_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("leak")

		exc_inh_check = If()
		exc_inh_check._if_.conditions.add("exc = '1'")
		exc_inh_check._if_.conditions.add("inh = '1'", "and")
		exc_inh_check._if_.body.add("next_state <= leak;")
		exc_inh_check._else_.body.add("next_state <= idle;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["leak"].\
				body.add(exc_inh_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("excite")

		exc_check = If()
		exc_check._if_.conditions.add("exc = '1'")
		exc_check._if_.body.add("next_state <= excite;")
		exc_check._else_.body.add("next_state <= idle;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["excite"].\
				body.add(exc_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list.\
				add("inhibit")

		inh_check = If()
		inh_check._if_.conditions.add("inh = '1'")
		inh_check._if_.body.add("next_state <= inhibit;")
		inh_check._else_.body.add("next_state <= idle;")

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].when_list["inhibit"].\
				body.add(inh_check)

		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].others.\
				body.add("next_state <= reset;")

		
		# Default values
		self.architecture.processes.add("output_evaluation")
		self.architecture.processes["output_evaluation"].\
				sensitivity_list.add("present_state")
		self.architecture.processes["output_evaluation"].\
				body.add("update_sel <= \"00\";")
		self.architecture.processes["output_evaluation"].\
				body.add("add_or_sub <= '0';")
		self.architecture.processes["output_evaluation"].\
				body.add("v_update <= '1';")
		self.architecture.processes["output_evaluation"].\
				body.add("v_en <= '1';")
		self.architecture.processes["output_evaluation"].\
				body.add("v_rst_n <= '1';")
		self.architecture.processes["output_evaluation"].\
				body.add("out_spike <= '0';")
		self.architecture.processes["output_evaluation"].\
				body.add("neuron_ready <= '0';")
		self.architecture.processes["output_evaluation"].\
				body.add("load_ready <= '0';")



		self.architecture.processes["output_evaluation"].\
				case_list.add("present_state")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("reset")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["reset"].\
				body.add("v_rst_n <= '0';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("load")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["load"].\
				body.add("load_ready <= '1';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("idle")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["idle"].\
				body.add("neuron_ready <= '1';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("init")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["init"].\
				body.add("v_rst_n <= '0';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("excite")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["excite"].\
				body.add("update_sel <= \"10\";")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["excite"].\
				body.add("v_en <= '1';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("inhibit")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["inhibit"].\
				body.add("update_sel <= \"11\";")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["inhibit"].\
				body.add("v_en <= '1';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("leak")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["leak"].\
				body.add("update_sel <= \"01\";")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["leak"].\
				body.add("v_en <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["leak"].\
				body.add("add_or_sub <= '1';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list.\
				add("fire")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["fire"].\
				body.add("v_update <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["fire"].\
				body.add("v_en <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["fire"].\
				body.add("out_spike <= '1';")

		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].others.\
				body.add("v_rst_n <= '0';")

		# Debug
		if(debug):
			self.debug = track_signals(self.architecture.signal,
					self.entity.name)

			for debug_port in self.debug:

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

	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")
