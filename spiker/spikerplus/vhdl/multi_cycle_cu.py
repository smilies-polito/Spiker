from .spiker_pkg import SpikerPackage
from .vhdl import track_signals, debug_component, sub_components, write_file_all

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If


class MultiCycleCU(VHDLblock):

	def __init__(self, learning = False, debug = False, debug_list = []):

		self.name = "multi_cycle_cu"

		self.learning = learning

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

		if self.learning:
			# The hand-coded Spiker-LL reference splits the single
			# all_ready input into the layers' ready AND and the
			# input-stream valid, checked in different FSM branches.
			self.entity.port.add(
					name		= "layers_ready",
					direction	= "in",
					port_type	= "std_logic")
			self.entity.port.add(
					name		= "input_ready",
					direction	= "in",
					port_type	= "std_logic")
		else:
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

		if self.learning:
			self.entity.port.add(
					name		= "timestep_end",
					direction	= "out",
					port_type	= "std_logic")
			self.entity.port.add(
					name		= "update_every_n_en",
					direction	= "out",
					port_type	= "std_logic")
			self.entity.port.add(
					name		= "vote_valid",
					direction	= "in",
					port_type	= "std_logic")
			self.entity.port.add(
					name		= "output_voter_en",
					direction	= "out",
					port_type	= "std_logic")
			self.entity.port.add(
					name		= "output_voter_vote",
					direction	= "out",
					port_type	= "std_logic")

		# Signals
		self.architecture.signal.add(
				name = "present_state",
				signal_type = "mc_states")

		self.architecture.signal.add(
				name = "next_state",
				signal_type = "mc_states")

		if self.learning:
			# Update-every-N gating sub-FSM, mirrored verbatim from
			# the hand-coded reference: after each sample (start),
			# skip the first two start_all pulses (pipeline
			# alignment), then let the datapath's counter count one
			# per timestep.
			self.architecture.declarationHeader.add(
				"type update_fsm_t is (S_START, S_WAIT, "
				"S_COUNTING);")
			self.architecture.declarationHeader.add(
				"signal state_q, state_d : update_fsm_t;")
			self.architecture.signal.add(
				name = "start_all_s",
				signal_type = "std_logic")

			self.architecture.bodyCodeHeader.add(
				"update_every_n_seq : process(clk, rst_n)\n"
				"    begin\n"
				"        if rst_n = '0' then\n"
				"            state_q         <= S_START;\n"
				"        elsif rising_edge(clk) then\n"
				"            state_q         <= state_d;\n"
				"        end if;\n"
				"    end process;")
			self.architecture.bodyCodeHeader.add(
				"update_every_n_comb : process(state_q, "
				"start_all_s, start)\n"
				"    begin\n"
				"        state_d         <= state_q;\n"
				"        case state_q is\n"
				"            when S_START =>\n"
				"                if start_all_s = '1' then\n"
				"                    state_d <= S_WAIT;\n"
				"                end if;\n"
				"            when S_WAIT =>\n"
				"                if start_all_s = '1' then\n"
				"                    state_d <= S_COUNTING;\n"
				"                end if;\n"
				"            when S_COUNTING =>\n"
				"                if start = '1' then\n"
				"                    state_d         <= S_START;\n"
				"                end if;\n"
				"        end case;\n"
				"    end process;")
			self.architecture.bodyCodeHeader.add(
				"update_every_n_en <= '1' when (state_q = "
				"S_COUNTING and start_all_s = '1') else '0';")
			self.architecture.bodyCodeHeader.add(
				"start_all <= start_all_s;")

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

		if self.learning:
			self.architecture.processes["state_evaluation"].\
					bodyHeader.add(
					"next_state <= present_state;")

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
		idle_ready = "layers_ready" if self.learning else "all_ready"
		all_ready_check = If()
		all_ready_check._if_.conditions.add(idle_ready + " = '1'")
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
		if self.learning:
			stop_check = If()
			stop_check._if_.conditions.add("stop = '1'")
			stop_check._if_.body.add("next_state <= vote;")
			stop_check._elsif_.add()
			stop_check._elsif_[0].conditions.add(
				"input_ready = '1'")
			stop_check._elsif_[0].body.add(
				"next_state <= network_update;")

			all_ready_check = If()
			all_ready_check._if_.conditions.add(
				"layers_ready = '1'")
			all_ready_check._if_.body.add(stop_check)
			all_ready_check._else_.body.add(
				"next_state <= update_wait;")
		else:
			stop_check = If()
			stop_check._if_.conditions.add("stop = '1'")
			stop_check._if_.body.add("next_state <= idle;")
			stop_check._else_.body.add(
				"next_state <= network_update;")

			all_ready_check = If()
			all_ready_check._if_.conditions.add("all_ready = '1'")
			all_ready_check._if_.body.add(stop_check)
			all_ready_check._else_.body.add(
				"next_state <= update_wait;")

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

		if self.learning:
			# Vote: hold until the output voter reports a valid
			# one-hot class.
			vote_check = If()
			vote_check._if_.conditions.add("vote_valid = '1'")
			vote_check._if_.body.add("next_state <= idle;")
			vote_check._else_.body.add("next_state <= vote;")
			self.architecture.processes["state_evaluation"].\
					case_list["present_state"].when_list[
					"vote"].body.add(vote_check)



		self.architecture.processes.add("output_evaluation")
		self.architecture.processes["output_evaluation"].\
				sensitivity_list.add("present_state")
		if self.learning:
			self.architecture.processes["output_evaluation"].\
					sensitivity_list.add("layers_ready")
			self.architecture.processes["output_evaluation"].\
					sensitivity_list.add("input_ready")
			self.architecture.processes["output_evaluation"].\
					sensitivity_list.add("stop")

		# In learning mode start_all is driven through start_all_s so
		# the update-every-N sub-FSM can observe it too.
		start_all_tgt = "start_all_s" if self.learning else "start_all"

		# Default values
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("ready <= '0';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add(start_all_tgt + " <= '0';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("cycles_cnt_en <= '0';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("cycles_cnt_rst_n <= '1';")
		self.architecture.processes["output_evaluation"].\
				bodyHeader.add("restart <= '0';")
		if self.learning:
			self.architecture.processes["output_evaluation"].\
					bodyHeader.add("timestep_end <= '0';")
			self.architecture.processes["output_evaluation"].\
					bodyHeader.add("output_voter_en <= '0';")
			self.architecture.processes["output_evaluation"].\
					bodyHeader.add("output_voter_vote <= '0';")

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
				body.add(start_all_tgt + " <= '0';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["update_wait"].\
				body.add("cycles_cnt_en <= '0';")
		if self.learning:
			# Pulse timestep_end when a timestep completes; also
			# enable the voter's counters on the final one.
			stop_out_check = If()
			stop_out_check._if_.conditions.add("stop = '1'")
			stop_out_check._if_.body.add("timestep_end <= '1';")
			stop_out_check._if_.body.add("output_voter_en <= '1';")
			stop_out_check._elsif_.add()
			stop_out_check._elsif_[0].conditions.add(
				"input_ready = '1'")
			stop_out_check._elsif_[0].body.add(
				"timestep_end <= '1';")

			layers_ready_out_check = If()
			layers_ready_out_check._if_.conditions.add(
				"layers_ready = '1'")
			layers_ready_out_check._if_.body.add(stop_out_check)
			self.architecture.processes["output_evaluation"].\
					case_list["present_state"].when_list[
					"update_wait"].body.add(
					layers_ready_out_check)

		# Network update
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["network_update"].\
				body.add(start_all_tgt + " <= '1';")
		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].when_list["network_update"].\
				body.add("cycles_cnt_en <= '1';")
		if self.learning:
			self.architecture.processes["output_evaluation"].\
					case_list["present_state"].when_list[
					"network_update"].body.add(
					"output_voter_en <= '1';")

			# Vote
			self.architecture.processes["output_evaluation"].\
					case_list["present_state"].when_list[
					"vote"].body.add(
					"output_voter_vote <= '1';")


		self.architecture.processes["output_evaluation"].\
				case_list["present_state"].others.body.add(
				"cycles_cnt_rst_n <= '0';")

				

		# Debug
		if debug:
			debug_component(self, debug_list)


	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)
