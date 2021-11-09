library ieee;
use ieee.std_logic_1164.all;


entity neuron_cu_tb is
end entity neuron_cu_tb;



architecture behaviour of neuron_cu_tb is



	-- input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal exp_exc_start	: std_logic;
	signal rest_inh_start	: std_logic;
	signal mask_neuron	: std_logic;
	signal input_spike	: std_logic;

	-- control input
	signal exceed_v_th	: std_logic;

	-- control output
	signal v_update_sel	: std_logic_vector(1 downto 0);
	signal v_or_v_th	: std_logic;
	signal add_or_sub	: std_logic;
	signal update_or_rest	: std_logic;
	signal v_reset_sel	: std_logic;
	signal v_th_en		: std_logic;
	signal v_en		: std_logic;

	-- output
	signal out_spike	: std_logic;




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

	-- clock
	clock_gen : process
	begin
		clk	<= '0';			-- falling edge i*12ns
		wait for 6 ns;			                            
		clk	<= '1';                 -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;



	-- reset
	reset_gen : process
	begin
		rst_n	<= '1';
		wait for 14 ns;
		rst_n	<= '0';
		wait for 3 ns;
		rst_n	<= '1';
		wait;
	end process reset_gen;



	-- exp_exc_start
	exp_exc_start_gen : process
	begin
		exp_exc_start	<= '0';		-- 0 ns
		wait for 26 ns;			
		exp_exc_start	<= '1';		-- 26 ns
		wait for 12 ns;			          
		exp_exc_start	<= '0';         -- 38 ns
		wait for 12 ns;			          
		exp_exc_start	<= '1';         -- 50 ns
		wait for 60 ns;			          
		exp_exc_start	<= '0';         -- 110 ns
		wait for 60 ns;
		exp_exc_start	<= '1';         -- 170 ns
		wait for 12 ns;
		exp_exc_start	<= '0';         -- 182 ns
		wait for 12 ns;
		exp_exc_start	<= '1';         -- 194 ns
		wait for 36 ns;
		exp_exc_start	<= '0';         -- 230 ns
		wait for 24 ns;
		exp_exc_start	<= '1';		-- 254 ns
		wait for 12 ns;
		exp_exc_start	<= '0';         -- 266 ns
		wait for 36 ns;
		exp_exc_start	<= '1';		-- 302 ns
		wait for 12 ns;
		exp_exc_start	<= '0';         -- 314 ns
		wait;
	end process exp_exc_start_gen;


	-- input_spike
	input_spike_gen: process
	begin
		input_spike	<= '0';		-- 0 ns	
		wait for 50 ns;			
		input_spike	<= '1';		-- 50 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 62 ns
		wait for 12 ns;			          
		input_spike	<= '1';         -- 74 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 86 ns
		wait for 12 ns;			          
		input_spike	<= '1';         -- 98 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 110 ns
		wait for 12 ns;			          
		input_spike	<= '1';         -- 122 ns
		wait for 24 ns;			          
		input_spike	<= '0';         -- 146 ns
		wait for 48 ns;
		input_spike	<= '1';         -- 194 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 206 ns
		wait for 12 ns;			          
		input_spike	<= '1';         -- 218 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 230 ns	
		wait for 48 ns;
		input_spike	<= '1';		-- 278 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 290 ns	
		wait;
	end process input_spike_gen;


	-- rest_inh_start
	rest_inh_start_gen : process
	begin
		rest_inh_start	<= '0';		-- 0 ns
		wait for 110 ns;		
		rest_inh_start	<= '1';		-- 110 ns
		wait for 36 ns;			          
		rest_inh_start	<= '0';         -- 146 ns
		wait for 132 ns;		
		rest_inh_start	<= '1';		-- 278 ns
		wait for 12 ns;
		rest_inh_start	<= '0';		-- 290 ns
		wait for 48 ns;
		rest_inh_start	<= '1';		-- 338 ns
		wait for 12 ns;
		rest_inh_start	<= '0';		-- 350 ns
		wait;
	end process rest_inh_start_gen;


	-- exceed_v_th
	exceed_v_th_gen : process
	begin
		exceed_v_th	<= '0';		-- 0 ns
		wait for 170 ns;
		exceed_v_th	<= '1';		-- 170 ns
		wait for 12 ns;
		exceed_v_th	<= '0';		-- 182 ns
		wait for 72 ns;
		exceed_v_th	<= '1';		-- 254 ns
		wait for 12 ns;
		exceed_v_th	<= '0';		-- 266 ns
		wait;
	end process exceed_v_th_gen;


	-- mask_neuron
	mask_neuron_gen : process
	begin
		mask_neuron	<= '0';		-- 0 ns
		wait for 134 ns;
		mask_neuron	<= '1';		-- 134 ns
		wait for 12 ns;
		mask_neuron	<= '0';		-- 146 ns
		wait for 48 ns;
		mask_neuron	<= '1';		-- 194 ns
		wait;
	end process mask_neuron_gen;













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
