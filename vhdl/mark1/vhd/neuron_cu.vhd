library ieee;
use ieee.std_logic_1164.all;


entity neuron_cu is

	port(
		-- input
		clk		: in std_logic;
		rst_n		: in std_logic;
		start		: in std_logic;
		start1		: in std_logic;
		start2		: in std_logic;
		rest_en		: in std_logic;
		mask_neuron	: in std_logic;
		input_spike	: in std_logic;

		-- input from datapath
		exceed_v_th	: in std_logic;

		-- control output
		no_update	: out std_logic;
		update_sel	: out std_logic_vector(1 downto 0);
		v_or_v_th	: out std_logic;
		add_or_sub	: out std_logic;
		v_th_update	: out std_logic;
		v_update	: out std_logic;
		v_th_en		: out std_logic;
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
		fire,
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
	state_evaluation	: process(present_state, start, start1, start2,
					exceed_v_th, input_spike, masked_spike,
					rest_en)


		procedure from_start2_on(
			
			-- input
			signal start2		: in std_logic;
			signal masked_spike	: in std_logic; 
			signal rest_en		: in std_logic;
			
			-- output
			signal next_state	: out states) is
		begin

			if start2 = '1'
			then
				-- inhibitory elaboration
				if masked_spike = '1'
				then
					next_state <= inh_spike;
				else
					next_state <= no_inh_spike;
				end if;
			else
				if rest_en = '1'
				then
					-- rest
					next_state <= rest;
				else
					if exceed_v_th = '1'
					then
						-- exponential decay
						next_state <= fire;
					else
						-- fire
						next_state <= exp_decay;
					end if;
				end if;
			end if;
		
		end procedure from_start2_on;
	
		
		
		procedure from_start1_on(
			
			-- input
			signal start1		: in std_logic;
			signal start2		: in std_logic;
			signal masked_spike	: in std_logic; 
			signal rest_en		: in std_logic;
			
			-- output
			signal next_state	: out states) is
		begin

			if start1 = '1'
			then
				-- inhibitory elaboration
				if masked_spike = '1'
				then
					next_state <= exc_spike;
				else
					next_state <= no_exc_spike;
				end if;
			else

				from_start2_on(
			
					-- input
					start2		=> start2,
					masked_spike	=> masked_spike,
					rest_en		=> rest_en,
					                   
					-- output          -- output
					next_state	=> next_state
				);

			end if;

		end procedure from_start1_on;


	begin


		-- default case
		next_state	<= reset;

		case present_state is
			
			-- reset
			when reset =>
				next_state <= idle;

			-- idle
			when idle =>
				if start = '1'
				then
					if exceed_v_th = '1'
					then
						next_state <= fire;
					else
						next_state <= exp_decay;
					end if;
				else

					next_state <= idle;

				end if;


			-- exp_decay
			when exp_decay =>

				from_start1_on(
			
					-- input
					start1		=> start1,
					start2		=> start2,
					masked_spike	=> masked_spike,
					rest_en		=> rest_en,

					-- output
					next_state	=> next_state	
				);

				
			-- fire
			when fire =>

				from_start1_on(
			
					-- input
					start1		=> start1,
					start2		=> start2,
					masked_spike	=> masked_spike,
					rest_en		=> rest_en,

					-- output
					next_state	=> next_state	
				);




			-- exc_spike
			when exc_spike =>
		
				from_start1_on(
			
					-- input
					start1		=> start1,
					start2		=> start2,
					masked_spike	=> masked_spike,
					rest_en		=> rest_en,

					-- output
					next_state	=> next_state	
				);



			-- no_exc_spike
			when no_exc_spike =>
				
				from_start1_on(
			
					-- input
					start1		=> start1,
					start2		=> start2,
					masked_spike	=> masked_spike,
					rest_en		=> rest_en,

					-- output
					next_state	=> next_state	
				);




			-- inh_spike
			when inh_spike =>
				
				from_start2_on(
			
					-- input
					start2		=> start2,
					masked_spike	=> masked_spike,
					rest_en		=> rest_en,
					                   
					-- output          -- output
					next_state	=> next_state
				);




			-- no_inh_spike
			when no_inh_spike =>
				
				from_start2_on(
			
					-- input
					start2		=> start2,
					masked_spike	=> masked_spike,
					rest_en		=> rest_en,
					                   
					-- output          -- output
					next_state	=> next_state
				);



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
		no_update	<= '0';
		update_sel	<= "00";
		v_or_v_th	<= '0';
		add_or_sub	<= '0';
		v_th_update	<= '0';
		v_update	<= '1';
		v_th_en		<= '0';
		v_en		<= '0';
		v_rst_n		<= '1';
		out_spike	<= '0';
		neuron_ready	<= '0';

		case present_state is

			-- reset
			when reset =>
				v_en 		<= '1';
				v_th_en		<= '1';
				v_update	<= '0';

			-- idle
			when idle =>
				neuron_ready	<= '1';

			-- exp_decay
			when exp_decay =>
				add_or_sub	<= '1';
				v_en		<= '1';
				
			-- fire
			when fire =>
				update_sel	<= "01";
				v_or_v_th	<= '1';
				v_th_update	<= '1';
				v_update	<= '0';
				v_th_en		<= '1';
				v_en		<= '1';
				out_spike	<= '1';

			-- exc_spike
			when exc_spike =>
				update_sel	<= "10";
				v_en		<= '1';

			-- no_exc_spike
			when no_exc_spike	=>
				no_update	<= '1';
				update_sel	<= "10";
				v_en		<= '0';
				
			-- inh_spike
			when inh_spike		=>
				update_sel	<= "11";
				v_en		<= '1';

			-- no_inh_spike
			when no_inh_spike	=>
				no_update	<= '1';
				update_sel	<= "11";
				v_en		<= '0';

			-- rest
			when rest =>
				v_rst_n		<= '0';

			-- default case
			when others =>
				v_en 		<= '1';
				v_th_en		<= '1';
				v_update	<= '0';

		end case;

	end process output_evaluation;
	

end architecture behaviour;
