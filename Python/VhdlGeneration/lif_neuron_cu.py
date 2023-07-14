import subprocess as sp

import path_config

from vhdl_block import VHDLblock


class LIFneuronCU(VHDLblock):

	def __init__(self):

		VHDLblock.__init__(self, entity_name = "neuron_cu")

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")


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

		self.architecture.customTypes.add(
				name 		= "states",
				c_type		= "Enumeration")

		self.architecture.customTypes["states"].add("reset")
		self.architecture.customTypes["states"].add("load")
		self.architecture.customTypes["states"].add("idle")
		self.architecture.customTypes["states"].add("init")
		self.architecture.customTypes["states"].add("excite")
		self.architecture.customTypes["states"].add("inhibit")
		self.architecture.customTypes["states"].add("fire")
		self.architecture.customTypes["states"].add("leak")

		self.architecture.signal.add(
				name = "present_state",
				signal_type = "states")

		self.architecture.signal.add(
				name = "next_state",
				signal_type = "states")

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
				if_list[0]._elsif_[0].conditions.add("clk = 1",
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
				body.add("next_state <= idle;")
		self.architecture.processes["state_evaluation"].\
				case_list["present_state"].others.\
				body.add("next_state <= reset;")






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


a = LIFneuronCU()

print(a.code())
