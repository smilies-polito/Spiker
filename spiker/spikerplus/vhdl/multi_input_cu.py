from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If

class MultiInputCU(VHDLblock):

	def __init__(self, debug = False, debug_list = []):
		
		self.name = "multi_input_cu"

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
				name            = "rst_n", 
				direction       = "in",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "restart", 
				direction       = "in",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "start", 
				direction       = "in",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "neurons_ready", 
				direction       = "in",
				port_type       = "std_logic")

		# Input from datapath
		self.entity.port.add(
				name            = "exc_yes", 
				direction       = "in",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "exc_stop", 
				direction       = "in",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "inh_yes", 
				direction       = "in",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "inh_stop", 
				direction       = "in",
				port_type       = "std_logic")

		# Output towards datapath
		self.entity.port.add(
				name            = "exc_cnt_en", 
				direction       = "out",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "exc_cnt_rst_n", 
				direction       = "out",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "inh_cnt_en",
				direction       = "out",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "inh_cnt_rst_n", 
				direction       = "out",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "exc", 
				direction       = "out",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "inh", 
				direction       = "out",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "spike_sample", 
				direction       = "out",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "spike_rst_n", 
				direction       = "out",
				port_type       = "std_logic")

		self.entity.port.add(
				name            = "neuron_restart", 
				direction       = "out",
				port_type       = "std_logic")

		# Output towards outside
		self.entity.port.add(
				name            = "ready", 
				direction       = "out",
				port_type       = "std_logic")


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
			       body.add("next_state <= idle_wait;")

		# Idle wait
		neurons_ready_check = If()
		neurons_ready_check._if_.conditions.add("neurons_ready = '1'")
		neurons_ready_check._if_.body.add("next_state <= idle;")
		neurons_ready_check._else_.body.add("next_state <= idle_wait;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
				"idle_wait"].body.add(neurons_ready_check)

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
		neurons_ready_check = If()
		neurons_ready_check._if_.conditions.add("neurons_ready = '1'")
		neurons_ready_check._if_.body.add("next_state <= init;")
		neurons_ready_check._else_.body.add("next_state <= init_wait;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
			       "init_wait"].body.add(neurons_ready_check)

		# Init
		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list["init"].\
			       body.add("next_state <= idle;")
			       
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
		neurons_ready_check = If()
		neurons_ready_check._if_.conditions.add("neurons_ready = '1'")
		neurons_ready_check._if_.body.add("next_state <= "
			"exc_update_full;")
		neurons_ready_check._else_.body.add("next_state <= "
			"exc_inh_wait;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
			       "exc_inh_wait"].body.add(neurons_ready_check)

		# Exc update full
		exc_stop_check = If()
		exc_stop_check._if_.conditions.add("exc_stop = '1'")
		exc_stop_check._if_.body.add("next_state <= inh_wait_full;")
		exc_stop_check._else_.body.add("next_state <= exc_update_full;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
			       "exc_update_full"].body.add(exc_stop_check)

		# Inh wait full
		neurons_ready_check = If()
		neurons_ready_check._if_.conditions.add("neurons_ready = '1'")
		neurons_ready_check._if_.body.add("next_state <= "
			"inh_update_full;")
		neurons_ready_check._else_.body.add("next_state <= "
			"inh_wait_full;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
			       "inh_wait_full"].body.add(neurons_ready_check)

		# Inh update full
		inh_stop_check = If()
		inh_stop_check._if_.conditions.add("inh_stop = '1'")
		inh_stop_check._if_.body.add("next_state <= idle;")
		inh_stop_check._else_.body.add("next_state <= inh_update_full;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
			       "inh_update_full"].body.add(inh_stop_check)

		# Exc wait
		neurons_ready_check = If()
		neurons_ready_check._if_.conditions.add("neurons_ready = '1'")
		neurons_ready_check._if_.body.add("next_state <= exc_update;")
		neurons_ready_check._else_.body.add("next_state <= exc_wait;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
			       "exc_wait"].body.add(neurons_ready_check)

		# Exc update 
		exc_stop_check = If()
		exc_stop_check._if_.conditions.add("exc_stop = '1'")
		exc_stop_check._if_.body.add("next_state <= idle;")
		exc_stop_check._else_.body.add("next_state <= exc_update;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
			       "exc_update"].body.add(exc_stop_check)

		# Inh wait
		neurons_ready_check = If()
		neurons_ready_check._if_.conditions.add("neurons_ready = '1'")
		neurons_ready_check._if_.body.add("next_state <= inh_update;")
		neurons_ready_check._else_.body.add("next_state <= inh_wait;")

		self.architecture.processes["state_evaluation"].\
			       case_list["present_state"].when_list[
			       "inh_wait"].body.add(neurons_ready_check)

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

		# Default values
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("exc_cnt_en <= '0';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("exc_cnt_rst_n <= '0';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("inh_cnt_en <= '0';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("inh_cnt_rst_n <= '0';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("exc <= '0';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("inh <= '0';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("spike_sample <= '0';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("spike_rst_n <= '1';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("neuron_restart <= '0';")
		self.architecture.processes["output_evaluation"].\
			       bodyHeader.add("ready <= '0';")

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
			       body.add("spike_rst_n <= '0';")

		# Init wait
		self.architecture.processes["output_evaluation"].\
			       case_list["present_state"].when_list[
			       "init_wait"].body.add("ready <= '0';")

		# Init
		self.architecture.processes["output_evaluation"].\
			       case_list["present_state"].when_list["init"].\
			       body.add("neuron_restart <= '1';")
			       
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
			       body.add("ready <= '0';")

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
			       "exc_cnt_rst_n <= '1';")
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
		self.architecture.processes["output_evaluation"].\
			       case_list["present_state"].when_list[
			       "inh_update_full"].body.add(
			       "inh_cnt_rst_n <= '1';")

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
			       "exc_cnt_rst_n <= '1';")
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
			       case_list["present_state"].when_list[
			       "inh_update"].body.add(
			       "inh_cnt_rst_n <= '1';")

		self.architecture.processes["output_evaluation"].\
			       case_list["present_state"].others.\
			       body.add("spike_rst_n <= '0';")

		# Debug
		if debug:
		       debug_component(self, debug_list)

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
