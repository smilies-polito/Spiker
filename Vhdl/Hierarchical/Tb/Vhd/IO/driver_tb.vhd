library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity driver_tb is
end entity driver_tb;

architecture test of driver_tb is

	-- Bit-width
	constant word_length		: integer := 64;
	constant load_bit_width		: integer := 4;
	constant data_bit_width		: integer := 36;
	constant addr_bit_width		: integer := 10;
	constant sel_bit_width		: integer := 10;
	constant cnt_out_bit_width	: integer := 16;

	-- Internal parameters
	constant N_inputs_tc_value	: integer := 2;
	constant N_neurons_tc_value	: integer := 1;
	constant N_cycles_tc_value	: integer := 30;
	constant v_reset_value		: integer := 5;
	constant inh_weight_value	: integer := -15;
	constant seed_value		: integer := 5;

	-- Initialization files
	constant weights_filename	: string  := "/home/alessio/Documents"&
		"/Poli/Dottorato/Progetti/Spiker/Vhdl/Hierarchical/Sim/IO"&
		"/weights.txt";

	constant v_th_filename		: string  := "/home/alessio/Documents"&
		"/Poli/Dottorato/Progetti/Spiker/Vhdl/Hierarchical/Sim/IO"&
		"/v_th.txt"; 

	constant inputs_filename	: string  := "/home/alessio/Documents"&
		"/Poli/Dottorato/Progetti/Spiker/Vhdl/Hierarchical/Sim/IO"&
		"/pixels.txt"; 
		

	-- input
	signal clk			: std_logic;
	signal driver_rst_n		: std_logic;
	signal go			: std_logic;
	signal N_inputs_cnt		: std_logic_vector(
						addr_bit_width-1
						downto 0);
	signal N_neurons_cnt		: std_logic_vector(
						addr_bit_width-1
						downto 0);
	signal input_word		: std_logic_vector(
						word_length-1 
						downto 0);

	-- output
	signal N_neurons_cnt_en		: std_logic;
	signal N_inputs_cnt_en		: std_logic;
	signal N_neurons_cnt_rst_n	: std_logic;
	signal N_inputs_cnt_rst_n	: std_logic;
	signal output_word		: std_logic_vector(word_length-1
						downto 0);

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

	signal N_neurons_tc		: std_logic;
	signal N_inputs_tc		: std_logic;


	type states is (
		reset,
		idle,
		load_N_inputs_tc,
		N_inputs_tc_wait,
		load_N_neurons_tc,
		N_neurons_tc_wait,
		load_N_cycles_tc,
		N_cycles_tc_wait,
		load_v_reset,
		v_reset_wait,
		load_inh_weight,
		inh_weight_wait,
		load_v_th,
		v_th_wait,
		load_weights,
		weights_wait,
		load_seed_lfsr,
		seed_lfsr_wait,
		wait_for_ready,
		read_counters,
		load_input_data,
		begin_execution
	);

	signal present_state, next_state	: states;


	component cnt is

		generic(
			bit_width		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			cnt_en		: in std_logic;
			cnt_rst_n	: in std_logic;

			-- output
			cnt_out		: out std_logic_vector(bit_width-1
						downto 0)		
		);

	end component cnt;

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

	output_word(word_length-1
		downto
		data_bit_width+
		addr_bit_width+
		sel_bit_width+
		load_bit_width+2)	<= (others => '0');



	
	-- clock
	clk_gen		: process
	begin
		clk <= '0';
		wait for 10 ns;
		clk <= '1';
		wait for 10 ns;
	end process clk_gen;

	-- reset (active low)
	driver_rst_n_gen	: process
	begin
		driver_rst_n <= '1';
		wait for 42 ns;
		driver_rst_n <= '0';
		wait for 10 ns;
		driver_rst_n <= '1';
		wait;
	end process driver_rst_n_gen;

	-- go
	go_gen	: process
	begin
		go <= '0';
		wait for 82 ns;
		go <= '1';
		wait for 1000 ns;
		go <= '1';
		wait;
	end process go_gen;

	-- ready
	ready_gen	: process
	begin
		input_word <= (others => '0');
		wait for 378 ns;
		input_word(input_word'length-1) <= '1';
		input_word(input_word'length-2 downto 0) <= (others => '0');
		wait for 20 ns;
		input_word <= (others => '0');
		wait;
	end process ready_gen;




	-- Neurons terminal counter
	N_neurons_tc_gen	: process(N_neurons_cnt)
	begin
		-- Default value
		N_neurons_tc <= '0';

		if N_neurons_cnt = std_logic_vector(
					to_unsigned(
					N_neurons_tc_value,
					N_neurons_cnt'length))
		then
			N_neurons_tc <= '1';
		end if;
		
	end process N_neurons_tc_gen;


	-- Inputs terminal counter
	N_inputs_tc_gen	: process(N_inputs_cnt)
	begin
		-- Default value
		N_inputs_tc <= '0';

		if N_inputs_cnt = std_logic_vector(
					to_unsigned(
					N_inputs_tc_value,
					N_inputs_cnt'length))
		then
			N_inputs_tc <= '1';
		end if;
		
	end process N_inputs_tc_gen;


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
	state_evaluation	: process(present_state, go, ready,
					N_neurons_tc, N_inputs_tc)
	begin

		case present_state is
			
			-- reset
			when reset =>
				next_state <= idle;

			-- idle
			when idle =>

				if go = '1'
				then
					next_state <= load_N_inputs_tc;
				else
					next_state <= idle;
				end if;

			

			-- load_N_inputs_tc
			when load_N_inputs_tc =>

				if go = '1'
				then
					next_state <= load_N_neurons_tc;
				else
					next_state <= N_inputs_tc_wait;
				end if;

			-- N_inputs_tc_wait
			when N_inputs_tc_wait =>

				if go = '1'
				then
					next_state <= load_N_neurons_tc;
				else
					next_state <= N_inputs_tc_wait;
				end if;


			-- load_N_neurons_tc
			when load_N_neurons_tc =>

				if go = '1'
				then
					next_state <= load_N_cycles_tc;
				else
					next_state <= N_neurons_tc_wait;
				end if;


			-- N_neurons_tc_wait
			when N_neurons_tc_wait =>

				if go = '1'
				then
					next_state <= load_N_cycles_tc;
				else
					next_state <= N_neurons_tc_wait;
				end if;


			-- load_N_cycles_tc
			when load_N_cycles_tc =>

				if go = '1'
				then
					next_state <= load_v_reset;
				else
					next_state <= N_cycles_tc_wait;
				end if;


			-- N_cycles_tc_wait
			when N_cycles_tc_wait =>

				if go = '1'
				then
					next_state <= load_v_reset;
				else
					next_state <= N_cycles_tc_wait;
				end if;


			-- load_v_reset
			when load_v_reset =>

				if go = '1'
				then
					next_state <= load_inh_weight;
				else
					next_state <= v_reset_wait;
				end if;

			-- v_reset_wait
			when v_reset_wait =>

				if go = '1'
				then
					next_state <= load_inh_weight;
				else
					next_state <= v_reset_wait;
				end if;



			-- load_inh_weight
			when load_inh_weight =>

				if go = '1'
				then
					next_state <= load_v_th;
				else
					next_state <= inh_weight_wait;
				end if;

				
			-- inh_weight_wait
			when inh_weight_wait =>

				if go = '1'
				then
					next_state <= load_v_th;
				else
					next_state <= inh_weight_wait;
				end if;



			-- load_v_th
			when load_v_th =>

				if N_neurons_tc = '0'
				then
					next_state <= load_v_th;

				elsif go = '1'
				then
					next_state <= load_weights;

				else
					next_state <= v_th_wait;
				end if;

			-- v_th_wait
			when v_th_wait =>

				if go = '1'
				then
					next_state <= load_weights;

				else
					next_state <= v_th_wait;
				end if;

			-- load_weights
			when load_weights =>

				if N_inputs_tc = '0'
				then
					next_state <= load_weights;

				elsif go = '1'
				then
					next_state <= load_seed_lfsr;

				else
					next_state <= weights_wait;
				end if;

			-- weights_wait
			when weights_wait =>

				if go = '1'
				then
					next_state <= load_seed_lfsr;

				else
					next_state <= weights_wait;
				end if;


			-- load_seed_lfsr
			when load_seed_lfsr =>

				if go = '1'
				then
					next_state <= wait_for_ready;
				else
					next_state <= seed_lfsr_wait;
				end if;

			-- seed_lfsr_wait
			when seed_lfsr_wait =>

				if go = '1'
				then
					next_state <= wait_for_ready;
				else
					next_state <= seed_lfsr_wait;
				end if;

			-- wait_for_ready
			when wait_for_ready =>

				if  go = '1' and ready = '1'
				then
					next_state <= read_counters;
				else
					next_state <= wait_for_ready;
				end if;

			-- read_counters
			when read_counters =>

				if N_neurons_tc = '1'
				then
					next_state <= load_input_data;
				else
					next_state <= read_counters;
				end if;

			-- load_input_data
			when load_input_data =>

				if N_inputs_tc = '1'
				then
					next_state <= begin_execution;
				else
					next_state <= load_input_data;
				end if;

			-- begin_execution
			when begin_execution =>
				
				next_state <= wait_for_ready;

			-- default case
			when others =>
				next_state <= reset;

		end case;

	end process state_evaluation;
	



	output_evaluation	: process(N_neurons_cnt, N_inputs_cnt,
					present_state)

		file weights_file	: text open read_mode is
			weights_filename;

		file v_th_file	: text open read_mode is
			v_th_filename;

		file inputs_file	: text open read_mode is
			inputs_filename;

		variable read_line	: line;

		variable data_var	: std_logic_vector(data'length-1
						downto 0);

	begin

		-- default values
		addr			<= (others => '0');
		load			<= (others => '1');
		data			<= (others => '0');
		sel			<= (others => '0');	
		N_inputs_cnt_en		<= '0';
		N_neurons_cnt_en	<= '0';
		N_inputs_cnt_rst_n	<= '1';
		N_neurons_cnt_rst_n	<= '1';
		start			<= '0';
		rst_n			<= '1';

		case present_state is

			-- reset
			when reset =>
				rst_n			<= '0';
				N_inputs_cnt_rst_n	<= '0';
				N_neurons_cnt_rst_n	<= '0';

			-- idle
			when idle =>
				rst_n	<= '0';

			-- load_N_inputs_tc
			when load_N_inputs_tc =>
				rst_n	<= '0';
				load	<= "0100";
				data	<= std_logic_vector(
						to_unsigned(
						N_inputs_tc_value,
						data'length));

			-- N_inputs_tc_wait
			when N_inputs_tc_wait =>
				rst_n	<= '0';

			-- load_N_neurons_tc
			when load_N_neurons_tc =>
				rst_n	<= '0';
				load	<= "0101";
				data	<= std_logic_vector(
						to_unsigned(
						N_neurons_tc_value,
						data'length));


			-- N_neurons_tc_wait
			when N_neurons_tc_wait =>
				rst_n	<= '0';

			-- load_N_cycles_tc
			when load_N_cycles_tc =>
				rst_n	<= '0';
				load	<= "0110";
				data	<= std_logic_vector(
						to_unsigned(
						N_cycles_tc_value,
						data'length));

			-- N_cycles_tc_wait
			when N_cycles_tc_wait =>
				rst_n	<= '0';

			-- load_v_reset
			when load_v_reset =>
				rst_n	<= '0';
				load	<= "0010";
				data	<= std_logic_vector(
						to_unsigned(
						v_reset_value,
						data'length));

			-- v_reset_wait
			when v_reset_wait =>
				rst_n	<= '0';

			-- load_inh_weight
			when load_inh_weight =>
				rst_n	<= '0';
				load	<= "0011";
				data	<= std_logic_vector(
						to_signed(
						inh_weight_value,
						data'length));
				
			-- inh_weight_wait
			when inh_weight_wait =>
				rst_n	<= '0';

			-- load_v_th
			when load_v_th =>

				rst_n			<= '0';
				N_neurons_cnt_en	<= '1';

				if clk = '1' and not endfile(v_th_file)
				then
					-- Read line from file
					readline(v_th_file, read_line);
					read(read_line, data_var);

					-- Associate line to data input
					data		<= data_var;

					load		<= "0001";
					addr		<= N_neurons_cnt;
				end if;

			-- v_th_wait
			when v_th_wait =>
				rst_n	<= '0';

			-- load_weights
			when load_weights =>

				rst_n		<= '0';
				N_inputs_cnt_en	<= '1';

				if not endfile(weights_file)
				then
					-- Read line from file
					readline(weights_file, read_line);
					read(read_line, data_var);

					-- Associate line to data input
					data		<= data_var;
					load		<= "0000";
					addr		<= N_inputs_cnt;
				end if;

			-- weights_wait
			when weights_wait =>
				rst_n	<= '0';

			-- load_seed_lfsr
			when load_seed_lfsr =>
				rst_n	<= '0';
				load	<= "0111";
				data	<= std_logic_vector(
						to_signed(
						seed_value,
						data'length));

				N_inputs_cnt_rst_n	<= '0';
				N_neurons_cnt_rst_n	<= '0';

			-- seed_lfsr_wait
			when seed_lfsr_wait =>
				rst_n	<= '0';

			-- wait_for_ready
			when wait_for_ready =>
				rst_n	<= '1';

			-- read_counters
			when read_counters =>
				rst_n			<= '1';
				N_neurons_cnt_en	<= '1';
				sel			<= N_neurons_cnt;

			-- load_input_data
			when load_input_data =>

				rst_n	<= '1';
				N_inputs_cnt_en	<= '1';

				if not endfile(inputs_file)
				then
					-- Read line from file
					readline(inputs_file, read_line);
					read(read_line, data_var);

					-- Associate line to data input
					data		<= data_var;
					rst_n		<= '0';
					load		<= "1000";
					addr		<= N_inputs_cnt;

				end if;

			-- begin_execution
			when begin_execution =>
				rst_n	<= '1';
				start	<= '1';

			-- default case
			when others =>
				rst_n			<= '0';
				N_inputs_cnt_rst_n	<= '0';
				N_neurons_cnt_rst_n	<= '0';
				

		end case;

	end process output_evaluation;



	N_neurons_counter	: cnt

		generic map(
			bit_width	=> addr_bit_width		
		)

		port map(
			-- input
			clk		=> clk,
			cnt_en		=> N_neurons_cnt_en,
			cnt_rst_n	=> N_neurons_cnt_rst_n,

			-- output
			cnt_out		=> N_neurons_cnt
		);


	N_inputs_counter	: cnt

		generic map(
			bit_width	=> addr_bit_width		
		)

		port map(
			-- input
			clk		=> clk,
			cnt_en		=> N_inputs_cnt_en,
			cnt_rst_n	=> N_inputs_cnt_rst_n,

			-- output
			cnt_out		=> N_inputs_cnt
		);

end architecture test;

