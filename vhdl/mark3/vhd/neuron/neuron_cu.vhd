library ieee;
use ieee.std_logic_1164.all;


entity neuron_cu is

	port(
		-- input
		clk		: in std_logic;
		rst_n		: in std_logic;
		start		: in std_logic;
		stop		: in std_logic;
		exc_or		: in std_logic;
		exc_stop	: in std_logic;
		inh_or		: in std_logic;
		inh_stop	: in std_logic;
		input_spike	: in std_logic;

		-- input from datapath
		exceed_v_th	: in std_logic;

		-- control output
		update_sel	: out std_logic_vector(1 downto 0);
		add_or_sub	: out std_logic;
		v_update	: out std_logic;
		v_en		: out std_logic;
		v_rst_n		: out std_logic;

		-- output
		out_spike	: out std_logic;
		neuron_ready	: out std_logic
	);

end entity neuron_cu;



architecture behaviour of neuron_cu is


	type states is (
		reset,
		idle,
		exp_decay,
		no_exc_spike,
		exc_spike,
		no_inh_spike,
		inh_spike,
		fire
	);

	signal present_state, next_state	: states;

begin



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
	state_evaluation	: process(present_state, start, stop, exc_or,
					inh_or, exc_stop, inh_stop, input_spike,
					exceed_v_th)
	begin


		-- default case
		-- next_state	<= reset;

		case present_state is
			
			-- reset
			when reset =>
				next_state <= idle;

			-- idle
			when idle =>

				if start = '1'
				then
					next_state <= exp_decay;
				else
					next_state <= idle;
				end if;
			

			-- exp_decay
			when exp_decay =>

				if stop = '1'
				then
					next_state <= idle;

				elsif exc_or = '1'
				then
					next_state <= no_exc_spike;

				elsif inh_or = '1'
				then
					next_state <= no_inh_spike;

				else
					next_state <= exp_decay;

				end if;

			-- no_exc_spike
			when no_exc_spike =>

				if exc_stop = '1'
				then
					if inh_or = '1'
					then
						next_state <= no_inh_spike;

					elsif exceed_v_th = '1'
					then
						next_state <= fire;

					else
						next_state <= exp_decay;
					end if;

				elsif input_spike = '1'
				then
					next_state <= exc_spike;

				else
					next_state <= no_exc_spike;
				end if;
				

			-- exc_spike
			when exc_spike =>

				if exc_stop = '1'
				then
					if inh_or = '1'
					then
						next_state <= no_inh_spike;

					elsif exceed_v_th = '1'
					then
						next_state <= fire;

					else
						next_state <= exp_decay;
					end if;

				elsif input_spike = '1'
				then
					next_state <= exc_spike;

				else
					next_state <= no_exc_spike;
				end if;


			-- no_inh_spike
			when no_inh_spike =>
				
				if inh_stop = '1'
				then
					if exceed_v_th = '1'
					then
						next_state <= fire;

					else
						next_state <= exp_decay;
					end if;

				elsif input_spike = '1'
				then
					next_state <= inh_spike;

				else
					next_state <= no_inh_spike;
				end if;



			-- inh_spike
			when inh_spike =>

				if inh_stop = '1'
				then
					if exceed_v_th = '1'
					then
						next_state <= fire;

					else
						next_state <= exp_decay;
					end if;

				elsif input_spike = '1'
				then
					next_state <= inh_spike;

				else
					next_state <= no_inh_spike;
				end if;

				

			

			-- fire
			when fire =>

				if stop = '1'
				then
					next_state <= idle;

				elsif exc_or = '1'
				then
					next_state <= no_exc_spike;

				else
					next_state <= no_inh_spike;

				end if;

			-- default case
			when others =>
				next_state <= reset;


		end case;

	end process state_evaluation;



	output_evaluation	: process(present_state)
	begin

		-- default values
		update_sel	<= "00";
		add_or_sub	<= '0';
		v_update	<= '1';
		v_en		<= '1';
		v_rst_n		<= '1';
		out_spike	<= '0';
		neuron_ready	<= '0';

		case present_state is

			-- reset
			when reset =>
				v_en 		<= '0';
				v_rst_n		<= '0';

			-- idle
			when idle =>
				neuron_ready	<= '1';
				v_en 		<= '0';
				v_rst_n		<= '0';

			-- exp_decay
			when exp_decay =>
				add_or_sub	<= '1';
				update_sel	<= "01";
				

			-- no_exc_spike
			when no_exc_spike	=>
				v_en		<= '0';

			-- exc_spike
			when exc_spike =>
				update_sel	<= "10";
				v_en		<= '1';

				
			-- no_inh_spike
			when no_inh_spike	=>
				v_en		<= '0';


			-- inh_spike
			when inh_spike		=>
				update_sel	<= "11";
				v_en		<= '1';


			-- fire
			when fire =>
				v_update	<= '0';
				out_spike	<= '1';

			-- default case
			when others =>
				v_en 		<= '0';
				v_rst_n		<= '0';

		end case;

	end process output_evaluation;
	

end architecture behaviour;
