library ieee;
use ieee.std_logic_1164.all;


entity neuron_cu_tb is
end entity neuron_cu_tb;



architecture behaviour of neuron_cu_tb is



	-- input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal start		: std_logic;
	signal start1		: std_logic;
	signal start2		: std_logic;
	signal rest_en		: std_logic;
	signal mask_neuron	: std_logic;
	signal input_spike	: std_logic;

	-- input from datapath
	signal exceed_v_th	: std_logic;

	-- control output
	signal update_sel	: std_logic_vector(1 downto 0);
	signal v_or_v_th	: std_logic;
	signal add_or_sub	: std_logic;
	signal v_th_update	: std_logic;
	signal v_update		: std_logic;
	signal v_th_en		: std_logic;
	signal v_en		: std_logic;
	signal v_rst_n		: std_logic;

	-- output
	signal out_spike	: std_logic;
	signal neuron_ready	: std_logic;




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

	-- clock
	clock_gen : process
	begin
		clk	<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;



	-- reset
	reset_gen : process
	begin
		rst_n	<= '1';		-- 0 ns
		wait for 14 ns;
		rst_n	<= '0';		-- 14 ns
		wait for 3 ns;
		rst_n	<= '1';		-- 17 ns
		wait;
	end process reset_gen;



	-- start
	start_gen : process
	begin
		start	<= '0';		-- 0 ns
		wait for 26 ns;
		start	<= '1';		-- 26 ns
		wait for 12 ns;
		start	<= '0';		-- 38 ns
		wait;
	end process start_gen;


	-- start1
	start1_gen : process
	begin
		start1	<= '0';		-- 0 ns
		wait for 38 ns;			
		start1	<= '1';		-- 38 ns
		wait for 24 ns;			          
		start1	<= '0';         -- 62 ns
		wait for 48 ns;			          
		start1	<= '1';         -- 110 ns
		wait for 12 ns;			          
		start1	<= '0';         -- 122 ns
		wait for 36 ns;
		start1	<= '1';         -- 158 ns
		wait for 12 ns;
		start1	<= '0';         -- 170 ns
		wait;
	end process start1_gen;



	-- start2
	start2_gen : process
	begin
		start2	<= '0';		-- 0 ns
		wait for 62 ns;		
		start2	<= '1';		-- 62 ns
		wait for 24 ns;			          
		start2	<= '0';         -- 86 ns
		wait for 48 ns;		
		start2	<= '1';		-- 134 ns
		wait for 12 ns;
		start2	<= '0';		-- 146 ns
		wait for 24 ns;
		start2	<= '1';		-- 170 ns
		wait for 12 ns;
		start2	<= '0';		-- 182 ns
		wait;
	end process start2_gen;


	-- input_spike
	input_spike_gen: process
	begin
		input_spike	<= '0';	-- 0 ns	
		wait for 38 ns;			
		input_spike	<= '1';	-- 38 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 50 ns
		wait for 12 ns;			          
		input_spike	<= '1'; -- 62 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 74 ns
		wait for 36 ns;			          
		input_spike	<= '1'; -- 110 ns
		wait for 48 ns;			          
		input_spike	<= '0'; -- 158 ns
		wait;
	end process input_spike_gen;


	-- mask_neuron
	mask_neuron_gen : process
	begin
		mask_neuron	<= '0';	-- 0 ns
		wait;
	end process mask_neuron_gen;


	-- exceed_v_th
	exceed_v_th_gen : process
	begin
		exceed_v_th	<= '0';	-- 0 ns
		wait for 146 ns;
		exceed_v_th	<= '1';	-- 146 ns
		wait for 12 ns;
		exceed_v_th	<= '0';	-- 158 ns
		wait for 24 ns;
		exceed_v_th	<= '1';	-- 182 ns
		wait for 12 ns;
		exceed_v_th	<= '0';	-- 194 ns
		wait;
	end process exceed_v_th_gen;



	-- rest_en
	rest_en_gen : process
	begin
		rest_en	<= '0';		-- 0 ns
		wait for 206 ns;
		rest_en	<= '1';		-- 206 ns
		wait for 12 ns;
		rest_en <= '0';		-- 218 ns
		wait;
	end process rest_en_gen;









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
				update_sel	<= "10";
				v_en		<= '0';
				
			-- inh_spike
			when inh_spike		=>
				update_sel	<= "11";
				v_en		<= '1';

			-- no_inh_spike
			when no_inh_spike	=>
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
