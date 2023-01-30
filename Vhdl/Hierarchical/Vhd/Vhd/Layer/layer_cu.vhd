library ieee;
use ieee.std_logic_1164.all;

entity layer_cu is

	port(
		-- input
		clk			: in std_logic;
		rst_n			: in std_logic;
		start			: in std_logic;

		-- signals from datapath
		exc_or			: in std_logic;		
		exc_stop		: in std_logic;		
		inh_or			: in std_logic;		
		inh_stop		: in std_logic;
		stop			: in std_logic;		

		-- towards datapath
		exc_en			: out std_logic;	
		anticipate_exc		: out std_logic;	
		inh_en			: out std_logic;	
		anticipate_inh		: out std_logic;	
		exc_cnt_en		: out std_logic;	
		exc_cnt_rst_n		: out std_logic;	
		inh_cnt_en		: out std_logic;	
		inh_cnt_rst_n		: out std_logic;	
		exc_or_inh_sel		: out std_logic;	
		inh			: out std_logic;	

		-- output
		cycles_cnt_rst_n	: out std_logic;	
		cycles_cnt_en		: out std_logic;	
		sample			: out std_logic;
		layer_ready		: out std_logic
	);

end entity layer_cu;


architecture behaviour of layer_cu is

	type states is(
		reset,
		idle,
		sample_spikes,
		exc_update,
		inh_update
	);

	signal present_state, next_state	: states;

begin


	-- state transition
	state_transition	: process(clk, rst_n)
	begin

		if rst_n = '0'
		then
			present_state	<= reset;

		elsif clk'event and clk = '1'
		then
			present_state	<= next_state;
		end if;

	end process state_transition;


	-- state evaluation
	state_evaluation	: process(present_state, start, stop, exc_or,
					exc_stop, inh_or, inh_stop)
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
					next_state <= sample_spikes;
				else
					next_state <= idle;
				end if;


			-- sample_spikes
			when sample_spikes =>

				if stop = '1'
				then
					next_state <= idle;
				else
					if exc_or = '1'
					then
						next_state <= exc_update;

					elsif inh_or = '1'
					then
						next_state <= inh_update;
					
					else
						next_state <= sample_spikes;
					end if;
				end if;


			-- exc_update
			when exc_update =>

				if exc_stop = '0'
				then
					next_state <= exc_update;

				elsif inh_or = '1'
				then
					next_state <= inh_update;

				else
					next_state <= sample_spikes;
				end if;


			-- inh_update
			when inh_update =>
				
				if inh_stop = '1'
				then
					next_state <= sample_spikes;
				else
					next_state <= inh_update;
				end if;


			when others =>

				next_state <= reset;		

		end case;

	end process state_evaluation;



	-- output evaluation
	output_evaluation	: process(present_state)
	begin

		-- default values
		exc_en			<= '0';
		anticipate_exc		<= '0';
		inh_en			<= '0';
		anticipate_inh		<= '0';
		exc_cnt_en		<= '0';
		exc_cnt_rst_n		<= '0';
		inh_cnt_en		<= '0';
		inh_cnt_rst_n		<= '0';
		exc_or_inh_sel		<= '0';
		inh			<= '0';
		layer_ready		<= '0';
		sample			<= '0';
		cycles_cnt_en		<= '0';
		cycles_cnt_rst_n	<= '1';


		case present_state is

			-- reset
			when reset =>

				cycles_cnt_rst_n	<= '0';

			-- idle
			when idle =>

				layer_ready		<= '1';
				cycles_cnt_rst_n	<= '0';

			-- sample_spikes
			when sample_spikes =>

				sample			<= '1';
				exc_en			<= '1';
				anticipate_exc		<= '1';
				inh_en			<= '1';
				anticipate_inh		<= '1';
				cycles_cnt_en		<= '1';

			-- exc_update
			when exc_update =>

				exc_cnt_en		<= '1';
				exc_cnt_rst_n		<= '1';


			-- inh_update
			when inh_update =>

				inh_cnt_en		<= '1';
				inh_cnt_rst_n		<= '1';
				exc_or_inh_sel		<= '1';
				inh			<= '1';

			when others =>

				cycles_cnt_rst_n	<= '0';

		end case;

	end process output_evaluation;


end architecture behaviour;
