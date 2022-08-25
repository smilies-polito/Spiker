library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity driver is
	generic(
		word_length		: integer := 36;
		load_bit_width		: integer := 4;
		data_bit_width		: integer := 36;
		addr_bit_width		: integer := 10;
		sel_bit_width		: integer := 10;
		cnt_out_bit_width	: integer := 16
 			
	);
	port(

		-- input
		clk		: in std_logic;
		driver_rst_n	: in std_logic;
		go		: in std_logic;

		input_word	: in std_logic_vector(word_length-1 downto 0);

		-- output
		output_word	: out std_logic_vector(word_length-1 downto 0)
	    
	);
end entity driver;


architecture behaviour of driver is

	signal addr			: std_logic_vector(
					     addr_bit_width-1
					     downto 0);
	signal load			: std_logic_vector(load_bit_width-1
					     downto 0);
	signal data			: std_logic_vector(
					     data_bit_width-1 
					     downto 0);
	signal sel			: std_logic_vector(
						sel_bit_width-1
						downto 0);
	signal start			: std_logic;
	signal rst_n			: std_logic;

	signal ready			: std_logic;
	signal cnt_out			: std_logic_vector(
						cnt_out_bit_width-1
						downto 0);


	type states is (
		reset,
		idle,
		load_N_inputs_tc,
		load_N_neurons_tc,
		load_N_cycles_tc,
		load_v_reset,
		load_inh_weight,
		load_v_th,
		v_th_wait,
		load_weights,
		load_seed_lfsr,
		read_counters,
		load_input_data,
		begin_execution
	);

	signal present_state, next_state	: states;

begin

	ready			<= input_word(input_word'length-1);
	cnt_out			<= input_word(cnt_out_bit_width-1 downto 0);

	output_word(
		data_bit_width-1 
		downto 
		0)			<= data;

	output_word(
		data_bit_width+
		addr_bit_width-1 
		downto 
		data_bit_width)		<= addr;

	output_word(
		data_bit_width+
		addr_bit_width+
		sel_bit_width-1
		downto 
		data_bit_width+
		addr_bit_width)		<= sel;

	output_word(
		data_bit_width+
		addr_bit_width+
		sel_bit_width+
		load_bit_width-1
		downto 
		data_bit_width+
		addr_bit_width+
		sel_bit_width)		<= load;

	output_word(
		data_bit_width+
		addr_bit_width+
		sel_bit_width+
		load_bit_width)		<= start;

	output_word(
		data_bit_width+
		addr_bit_width+
		sel_bit_width+
		load_bit_width+1)	<= rst_n;


	-- state transition
	state_transition	: process(clk, driver_rst_n)
	begin

		-- active reset
		if driver_rst_n = '0'
		then
			present_state <= reset;

		-- inactive reset
		elsif clk'event and clk = '1'
		then
			present_state <= next_state;
		end if;

	end process state_transition;




	-- state evaluation
	state_evaluation	: process(present_state, go, ready)


	begin

		case present_state is
			
			-- reset
			when reset =>
				next_state <= idle;

			-- idle
			when idle =>
			

			-- load_N_inputs_tc
			when load_N_inputs_tc =>

				if go = '1'
				then
					next_state <= load_N_neurons_tc;
				else
					next_state <= load_N_inputs_tc;
				end if;

			-- load_N_neurons_tc
			when load_N_neurons_tc =>

				if go = '1'
				then
					next_state <= load_N_cycles_tc;
				else
					next_state <= load_N_neurons_tc;
				end if;


			-- load_N_cycles_tc
			when load_N_cycles_tc =>

				if go = '1'
				then
					next_state <= load_v_reset;
				else
					next_state <= load_N_cycles_tc;
				end if;


			-- load_v_reset
			when load_v_reset =>

				if go = '1'
				then
					next_state <= load_inh_weight;
				else
					next_state <= load_v_reset;
				end if;


			-- load_inh_weight
			when load_inh_weight =>

				if go = '1'
				then
					next_state <= load_v_th;
				else
					next_state <= load_inh_weight;
				end if;


			-- load_v_th
			when load_v_th =>

				if go = '1'
				then
					next_state <= load_N_cycles_tc;

				elsif go = '1'
				then
					next_state <= load_weights;

				else
					next_state <= v_th_wait;
				end if;


			-- default case
			when others =>
				next_state <= reset;


		end case;

	end process state_evaluation;



--	output_evaluation	: process(present_state)
--	begin
--
--		-- Default values
--		data	<= (others => '0');
--		addr	<= (others => '0');
--		sel	<= (others => '0')
--		load	<= (others => '1');
--		start	<= '0';
--		rst_n	<= '1';
--
--		case present_state is
--
--			-- reset
--			when reset =>
--				v_en 		<= '0';
--				v_rst_n		<= '0';
--
--			-- idle
--			when idle =>
--				neuron_ready	<= '1';
--				v_en 		<= '0';
--				v_rst_n		<= '0';
--
--			-- exp_decay
--			when exp_decay =>
--				add_or_sub	<= '1';
--				update_sel	<= "01";
--				
--
--			-- no_exc_spike
--			when no_exc_spike	=>
--				v_en		<= '0';
--
--			-- exc_spike
--			when exc_spike =>
--				update_sel	<= "10";
--				v_en		<= '1';
--
--				
--			-- no_inh_spike
--			when no_inh_spike	=>
--				v_en		<= '0';
--
--
--			-- inh_spike
--			when inh_spike		=>
--				update_sel	<= "11";
--				v_en		<= '1';
--
--
--			-- fire
--			when fire =>
--				v_update	<= '0';
--				out_spike	<= '1';
--
--			-- default case
--			when others =>
--				v_en 		<= '0';
--				v_rst_n		<= '0';
--
--		end case;
--
--	end process output_evaluation;

end architecture behaviour;
