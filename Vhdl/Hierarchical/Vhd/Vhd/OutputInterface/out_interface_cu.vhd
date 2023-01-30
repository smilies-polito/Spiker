library ieee;
use ieee.std_logic_1164.all;


entity out_interface_cu is

	port(
		-- input
		clk		: in std_logic;
		rst_n		: in std_logic;
		start		: in std_logic;
		stop		: in std_logic;

		-- output
		elaborate	: out std_logic;
		cnt_rst_n	: out std_logic
	);

end entity out_interface_cu;



architecture behaviour of out_interface_cu is


	type states is (
		reset,
		idle,
		init,
		elaboration
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
	state_evaluation	: process(present_state, start, stop)
	begin

		case present_state is
			
			-- reset
			when reset =>
				next_state <= idle;

			-- idle
			when idle =>

				if start = '1'
				then
					next_state <= init;
				else
					next_state <= idle;
				end if;

			-- init
			when init =>
				next_state <= elaboration;

			-- elaboration
			when elaboration =>

				if stop = '1'
				then
					next_state <= idle;
				else
					next_state <= elaboration;
				end if;

			-- default case
			when others =>
				next_state <= reset;

		end case;

	end process state_evaluation;



	output_evaluation	: process(present_state)
	begin

		-- default values
		cnt_rst_n <= '1';
		elaborate <= '0';

		case present_state is

			-- reset
			when reset =>
				cnt_rst_n <= '0';

			-- idle
			when idle =>
				cnt_rst_n <= '1';
				elaborate <= '0';

			-- init
			when init =>
				cnt_rst_n <= '0';

			-- elaboration
			when elaboration=>
				elaborate <= '1';
			
			-- default case
			when others =>
				cnt_rst_n <= '0';
				
		end case;

	end process output_evaluation;
	

end architecture behaviour;

