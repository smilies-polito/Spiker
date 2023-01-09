library ieee;
use ieee.std_logic_1164.all;

entity layer_cu is

	port(
		-- input
		clk			: in std_logic;
		rst_n			: in std_logic;
		start			: in std_logic;

		-- signals from datapath
		stop			: in std_logic;		
		exc_or			: in std_logic;		
		exc_stop		: in std_logic;		
		inh_or			: in std_logic;		
		inh_stop		: in std_logic;

		-- towards datapath
		exc_en			: out std_logic;	
		anticipate_exc		: out std_logic;	
		inh_en			: out std_logic;	
		anticipate_inh		: out std_logic;	
		exc_cnt_rst_n		: out std_logic;	
		exc_cnt_en		: out std_logic;	
		inh_cnt_rst_n		: out std_logic;	
		inh_cnt_en		: out std_logic;	
		cycles_cnt_rst_n	: out std_logic;	
		cycles_cnt_en		: out std_logic;	
		exc_or_inh_sel		: out std_logic;	
		inh_elaboration		: out std_logic;	
		rest_en			: out std_logic;	
		mask1			: out std_logic;	
		mask2			: out std_logic;

		-- output
		sample			: out std_logic;
		snn_ready		: out std_logic
	);

end entity layer_cu;


architecture behaviour of layer_cu is

	type states is(
		reset,
		idle,
		sample_spikes,
		exc_update,
		exc_end,
		inh_update,
		inh_end,
		rest		
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

		procedure from_inh_stop_on(

			-- input
			signal inh_stop		: in std_logic;

			-- output
			signal next_state	: out states) is
		begin

			if inh_stop = '1'
			then
				next_state <= inh_end;
			else
				next_state <= inh_update;
			end if;

		end procedure from_inh_stop_on;


		procedure from_inh_or_on(
			
			-- input
			signal inh_or		: in std_logic;
			signal inh_stop		: in std_logic;

			-- output
			signal next_state	: out states) is
		begin

			if inh_or ='1'
			then
				from_inh_stop_on(

					-- input
					inh_stop	=> inh_stop,

					-- output
					next_state	=> next_state		
				);
			else
				next_state <= sample_spikes;
			end if;

		end procedure from_inh_or_on;


		procedure from_exc_stop_on(

			-- input
			signal exc_stop		: in std_logic;

			-- output
			signal next_state	: out states) is
		begin

			if exc_stop = '1'
			then
				next_state <= exc_end;
			else
				next_state <= exc_update;
			end if;

		end procedure from_exc_stop_on;


		procedure from_exc_or_on(
			
			-- input
			signal exc_or		: in std_logic;
			signal exc_stop		: in std_logic;
			signal inh_or		: in std_logic;
			signal inh_stop		: in std_logic;

			-- output
			signal next_state	: out states) is
		begin

			if exc_or ='1'
			then
				from_exc_stop_on(

					-- input
					exc_stop	=> exc_stop,

					-- output
					next_state	=> next_state		
				);
			else
				
				from_inh_or_on(
			
					-- input
					inh_or		=> inh_or,
					inh_stop	=> inh_stop,

					-- output
					next_state	=> next_state
				);
			end if;

		end procedure from_exc_or_on;


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
					next_state <= rest;
				else
					from_exc_or_on(
						-- input
						exc_or		=> exc_or,
						exc_stop	=> exc_stop,
						inh_or		=> inh_or,
						inh_stop	=> inh_stop,
                                                                           
						-- output          
						next_state	=> next_state
	
					);
				end if;

			-- exc_update
			when exc_update =>

				from_exc_stop_on(

					-- input
					exc_stop	=> exc_stop,

					-- output
					next_state	=> next_state		
				);		

			-- exc_end
			when exc_end =>		

				from_inh_or_on(
			
					-- input
					inh_or		=> inh_or,
					inh_stop	=> inh_stop,

					-- output
					next_state	=> next_state
				);

			-- inh_update
			when inh_update =>

				from_inh_stop_on(

					-- input
					inh_stop	=> inh_stop,

					-- output
					next_state	=> next_state		
				);		

			-- inh_end
			when inh_end =>

				next_state <= sample_spikes;		

			-- rest
			when rest =>

				next_state <= idle;

			when others =>

				next_state <= reset;

		end case;

	end process state_evaluation;




	output_evaluation	: process(present_state)
	begin

		-- default values
		sample			<= '0';
		snn_ready		<= '0';
		exc_en			<= '0';
		anticipate_exc		<= '0';
		inh_en			<= '0';
		anticipate_inh		<= '0';
		exc_cnt_en		<= '0';
		exc_cnt_rst_n		<= '1';
		inh_cnt_en		<= '0';
		inh_cnt_rst_n		<= '1';
		exc_or_inh_sel		<= '0';
		mask1			<= '0';
		mask2			<= '0';
		cycles_cnt_en		<= '0';
		cycles_cnt_rst_n	<= '1';
		rest_en			<= '0';
		inh_elaboration		<= '0';


		case present_state is

			-- reset
			when reset =>

				exc_cnt_rst_n		<= '0';
				inh_cnt_rst_n		<= '0';
				cycles_cnt_rst_n	<= '0';
				mask1			<= '0';
				mask2			<= '0';

			-- idle
			when idle =>

				snn_ready		<= '1';
				mask1			<= '0';
				mask2			<= '0';

			-- sample_spikes
			when sample_spikes =>

				sample			<= '1';
				exc_en			<= '1';
				inh_en			<= '1';
				cycles_cnt_en		<= '1';

			-- exc_update
			when exc_update =>

				mask2			<= '0';
				exc_cnt_en		<= '1';

			-- exc_end
			when exc_end =>

				mask1			<= '0';
				mask2			<= '0';
				exc_cnt_rst_n		<= '0';

			-- inh_update
			when inh_update =>

				mask1			<= '0';
				inh_cnt_en		<= '1';
				inh_elaboration		<= '1';

			-- inh_end
			when inh_end =>

				mask1			<= '0';
				mask2			<= '0';
				inh_cnt_rst_n		<= '0';
				inh_elaboration		<= '1';

			-- rest
			when rest =>

				rest_en			<= '1';
				cycles_cnt_rst_n	<= '0';

			when others =>

				exc_cnt_rst_n		<= '0';
				inh_cnt_rst_n		<= '0';
				cycles_cnt_rst_n	<= '0';
				mask1			<= '0';
				mask2			<= '0';

		end case;

	end process output_evaluation;


end architecture behaviour;
