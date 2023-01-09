library ieee;
use ieee.std_logic_1164.all;


entity neuron_cu is

	port(
		-- input
		clk		: in std_logic;
		rst_n		: in std_logic;
		exp_exc_start	: in std_logic;
		rest_inh_start	: in std_logic;
		mask_neuron	: in std_logic;
		input_spike	: in std_logic;

		-- input from datapath
		exceed_v_th	: in std_logic;

		-- control output
		v_update_sel	: out std_logic_vector(1 downto 0);
		v_or_v_th	: out std_logic;
		add_or_sub	: out std_logic;
		update_or_rest	: out std_logic;
		v_reset_sel	: out std_logic;
		v_th_en		: out std_logic;
		v_en		: out std_logic;

		-- output
		out_spike	: out std_logic
	);

end entity neuron_cu;



architecture behaviour of neuron_cu is


	type states is (
		reset,
		idle,
		exp_decay1,
		exp_decay2,
		fire,
		pause,
		exc_spike,
		no_exc_spike,
		inh_spike,
		no_inh_spike,
		rest
	);

	signal present_state, next_state	: states;
	signal masked_spike			: std_logic;

begin


	masked_spike <= input_spike and not mask_neuron;



	-- state transition
	state_transition	: process(clk, rst_n)
	begin

		-- active reset
		if rst_n = '0'
		then
			present_state <= reset;

		-- inactive reset
		elsif clk'event and clk = '1'
		then
			present_state <= next_state;
		end if;

	end process state_transition;




	-- state evaluation
	state_evaluation	: process(present_state, exp_exc_start,
					exceed_v_th, rest_inh_start, 
					input_spike, masked_spike)
	begin

		-- default case
		next_state	<= reset;

		case present_state is
			
			-- reset
			when reset =>
				next_state <= idle;

			-- idle
			when idle =>
				if exp_exc_start = '1'
				then
					if exceed_v_th = '1'
					then
						next_state <= fire;
					else
						next_state <= exp_decay1;
					end if;
				else
					if rest_inh_start = '1'
					then
						next_state <= rest;
					else
						next_state <= idle;
					end if;
				end if;

			-- exp_decay1
			when exp_decay1 =>
				next_state <= exp_decay2;
				
			-- exp_decay2
			when exp_decay2 =>
				if exp_exc_start = '1'
				then
					if input_spike = '1'
					then
						next_state <= exc_spike;
					else
						next_state <= no_exc_spike;
					end if;
				else
					if rest_inh_start = '1'
					then
						if input_spike = '1'
						then
							next_state <= inh_spike;
						else
							next_state <= no_inh_spike;
						end if;
					else
						next_state <= idle;
					end if;
				end if;

			-- fire
			when fire =>
				next_state <= pause;

			-- pause
			when pause =>
				if exp_exc_start = '1'
				then
					if input_spike = '1'
					then
						next_state <= exc_spike;
					else
						next_state <= no_exc_spike;
					end if;
				else
					if rest_inh_start = '1'
					then
						if input_spike = '1'
						then
							next_state <= inh_spike;
						else
							next_state <= no_inh_spike;
						end if;
					else
						next_state <= idle;
					end if;
				end if;

			-- exc_spike
			when exc_spike =>
				if exp_exc_start = '1'
				then
					if input_spike = '1'
					then
						next_state <= exc_spike;
					else
						next_state <= no_exc_spike;
					end if;
				else
					if rest_inh_start = '1'
					then
						if input_spike = '1'
						then
							next_state <= inh_spike;
						else
							next_state <= no_inh_spike;
						end if;
					else
						next_state <= idle;
					end if;
				end if;

			-- no_exc_spike
			when no_exc_spike =>
				if exp_exc_start = '1'
				then
					if input_spike = '1'
					then
						next_state <= exc_spike;
					else
						next_state <= no_exc_spike;
					end if;
				else
					if rest_inh_start = '1'
					then
						if input_spike = '1'
						then
							next_state <= inh_spike;
						else
							next_state <= no_inh_spike;
						end if;
					else
						next_state <= idle;
					end if;
				end if;

			-- inh_spike
			when inh_spike =>
				if rest_inh_start = '1'
				then
					if masked_spike = '1'
					then
						next_state <= inh_spike;
					else
						next_state <= no_inh_spike;
					end if;
				else
					next_state <= idle;
				end if;

			-- no_inh_spike
			when no_inh_spike =>
				if rest_inh_start = '1'
				then
					if masked_spike = '1'
					then
						next_state <= inh_spike;
					else
						next_state <= no_inh_spike;
					end if;
				else
					next_state <= idle;
				end if;

			-- rest
			when rest =>
				next_state <= idle;
			-- default case
			when others =>
				next_state <= reset;

		end case;

	end process state_evaluation;





	output_evaluation	: process(present_state)
	begin

		-- default values
		v_update_sel	<= "00";
		v_or_v_th	<= '0';
		add_or_sub	<= '0';
		update_or_rest	<= '1';
		v_reset_sel	<= '0';
		v_th_en		<= '0';
		v_en		<= '0';
		out_spike	<= '0';

		case present_state is

			-- reset
			when reset =>
				v_en 		<= '1';
				v_th_en		<= '1';
				v_reset_sel	<= '1';
				update_or_rest	<= '0';

			-- idle
			when idle =>
				v_en		<= '0';
				v_th_en		<= '0';

			-- exp_decay1
			when exp_decay1 =>
				v_update_sel	<= "00";
				add_or_sub	<= '1';
				v_en		<= '1';

			-- exp_decay2
			when exp_decay2	=>
				v_update_sel	<= "01";
				add_or_sub	<= '0';
				v_en		<= '1';
				
			-- fire
			when fire =>
				v_or_v_th	<= '1';
				add_or_sub	<= '0';
				v_reset_sel	<= '1';
				v_th_en		<= '1';
				v_en		<= '1';
				out_spike	<= '1';

			-- pause
			when pause =>
				v_en		<= '0';
				v_th_en		<= '0';

			-- exc_spike
			when exc_spike =>
				v_update_sel	<= "10";
				add_or_sub	<= '0';
				v_en		<= '1';

			-- no_exc_spike
			when no_exc_spike	=>
				v_update_sel	<= "10";
				add_or_sub	<= '0';
				v_en		<= '0';
				
			-- inh_spike
			when inh_spike		=>
				v_update_sel	<= "11";
				add_or_sub	<= '0';
				v_en		<= '1';

			-- no_inh_spike
			when no_inh_spike	=>
				v_update_sel	<= "11";
				add_or_sub	<= '0';
				v_en		<= '0';

			-- rest
			when rest =>
				update_or_rest	<= '0';
				v_en		<= '1';

			-- default case
			when others =>
				v_update_sel	<= "00";
				v_or_v_th	<= '0';
				add_or_sub	<= '0';
				update_or_rest	<= '1';
				v_reset_sel	<= '0';
				v_th_en		<= '0';
				v_en		<= '0';
				out_spike	<= '0';

		end case;

	end process output_evaluation;
	

end architecture behaviour;
