library ieee;
use ieee.std_logic_1164.all;


entity neuron_cu is

	port(
		-- input
		clk		: in std_logic;
		rst_n		: in std_logic;
		restart		: in std_logic;
		load_end	: in std_logic;
		exc		: in std_logic;
		inh		: in std_logic;

		-- input from datapath
		exceed_v_th	: in std_logic;

		-- control output
		update_sel	: out std_logic_vector(1 downto 0);
		add_or_sub	: out std_logic;
		v_update	: out std_logic;
		v_en		: out std_logic;
		v_rst_n		: out std_logic;

		-- output
		load_ready	: out std_logic;
		neuron_ready	: out std_logic;
		out_spike	: out std_logic
	);

end entity neuron_cu;



architecture behaviour of neuron_cu is


	type states is (
		reset,
		load,
		idle,
		init,
		excite,
		inhibit,
		leak,
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
	state_evaluation	: process(present_state, load_end, restart, exc,
					inh, exceed_v_th)
	begin

		case present_state is
			
			-- reset
			when reset =>
				next_state <= load;


			-- load
			when load =>

				if load_end = '1'
				then
					next_state <= idle;
				else
					next_state <= load;
				end if;

			-- idle
			when idle =>

				if restart = '1'
				then
					next_state <= init;

				elsif exc = '1'
				then

					if inh = '0'
					then
						next_state <= excite;

					elsif exceed_v_th = '0'
					then
						next_state <= leak;

					else
						next_state <= fire;

					end if;

				elsif inh = '1'
				then
					next_state <= inhibit;

				else
					next_state <= idle;
				end if;
			

			-- excite
			when excite =>

				if exc = '1'
				then
					next_state <= excite;

				else
					next_state <= idle;

				end if;

			-- inhibit
			when inhibit =>

				if inh = '1'
				then
					next_state <= inhibit;

				else
					next_state <= idle;

				end if;


			-- fire
			when fire =>

				if exc = '1' and inh = '1'
				then
					next_state <= leak;

				elsif 
					next_state <= idle;

				end if;

			-- leak
			when leak =>

				if exc = '1' and inh = '1'
				then
					next_state <= leak;

				elsif 
					next_state <= idle;

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
		v_update	<= '0';
		v_en		<= '0';
		v_rst_n		<= '1';
		neuron_ready	<= '0';
		load_ready	<= '0';
		out_spike	<= '0';

		case present_state is

			-- reset
			when reset =>
				v_rst_n		<= '0';

			when load =>
				load_ready	<= '1';

			-- idle
			when idle =>
				neuron_ready	<= '1';

			-- init
			when init =>
				v_rst_n		<= '0';

			-- excite
			when excite =>
				update_sel	<= "10";
				v_en		<= '1';

			-- inhibit 
			when inhibit =>
				update_sel	<= "11";
				v_en		<= '1';
				

			-- leak
			when leak =>
				add_or_sub	<= '1';
				update_sel	<= "01";
				v_en		<= '1';

			-- fire
			when fire =>
				v_update	<= '1';
				v_en		<= '1';
				out_spike	<= '1';

			-- default case
			when others =>
				v_rst_n		<= '0';

		end case;

	end process output_evaluation;
	

end architecture behaviour;
