library ieee;
library std;
use std.env.finish;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity spiker_tb is
end entity spiker_tb;

architecture test of spiker_tb is

	-- Network parameters
	constant v_reset_int		: integer := 5*2**3; 	  
	constant inh_weight_int	 	: integer := 0; 
	constant seed_int		: integer := 5;

	-- Bit-widths
	constant neuron_bit_width	: integer := 16;
	constant weights_bit_width	: integer := 5;
	constant bram_word_length	: integer := 36;
	constant bram_addr_length	: integer := 10;
	constant bram_sel_length	: integer := 6;
	constant bram_we_length		: integer := 4;
	constant input_data_bit_width	: integer := 8;
	constant lfsr_bit_width		: integer := 16;
	constant cnt_out_bit_width	: integer := 16;

	-- Network shape
	constant inputs_addr_bit_width	: integer := 10;
	constant neurons_addr_bit_width	: integer := 9;

	-- Must be 1 bit longer than what required to count to N_cycles
	constant cycles_cnt_bit_width	: integer := 12;

	-- Bram parameters
	constant N_bram			: integer := 58;
	constant N_weights_per_word	: integer := 7;

	-- Structure parameters
	constant N_inputs		: integer := 784;
	constant N_neurons		: integer := 400;
	constant N_cycles		: integer := 30;

	-- Internal parameters
	constant shift			: integer := 10;


	constant weights_filename	: string	:= "/home/alessio/Documents/"&
		"Poli/Dottorato/Progetti/spiker/vhdl/mark3/sim/hyperparameters/"&
		"weights.mem";

	constant thresholds_filename	: string	:= "/home/alessio/Documents/"&
		"Poli/Dottorato/Progetti/spiker/vhdl/mark3/sim/hyperparameters/"&
		"thresholds.init";

	constant inputs_filename	: string	:= "/home/alessio/Documents/"&
		"Poli/Dottorato/Progetti/spiker/vhdl/mark3/sim/inputOutput/"&
		"inputImage.txt";

	constant output_filename	: string	:= "/home/alessio/Documents/"&
		"Poli/Dottorato/Progetti/spiker/vhdl/mark3/sim/inputOutput/"&
		"cntOut.txt";


	-- Spiker inputs 
	signal clk			: std_logic;
	signal rst_n			: std_logic;	
	signal init_lfsr		: std_logic;
	signal seed			: unsigned(lfsr_bit_width-1 downto 0); 
	signal start			: std_logic;	
	signal init_v_th		: std_logic;
	signal v_th_addr		: std_logic_vector(
					     neurons_addr_bit_width 
					     downto 0);
	signal input_data		: unsigned(N_inputs*
					     lfsr_bit_width-1 
					     downto 0);
	signal v_th_value		: signed(neuron_bit_width-1 
					     downto 0);		
	signal v_reset			: signed(neuron_bit_width-1 
					     downto 0);	
	signal inh_weight		: signed(neuron_bit_width-1
					     downto 0);		
	signal N_inputs_tc		: std_logic_vector (
					     inputs_addr_bit_width downto 0);
	signal N_neurons_tc		: std_logic_vector(
					     neurons_addr_bit_width 
					     downto 0);
	signal N_cycles_tc		: std_logic_vector(
					     cycles_cnt_bit_width-1 
					     downto 0);
	signal di			: std_logic_vector(bram_word_length-1 
					     downto 0);
	signal rden			: std_logic;
	signal wren			: std_logic;
	signal wraddr			: std_logic_vector(bram_addr_length-1 
					     downto 0);
	signal bram_sel			: std_logic_vector(bram_sel_length-1 
						downto 0);

	-- Spiker output
	signal ready			: std_logic;
	signal cnt_out			: std_logic_vector(N_neurons*
						cnt_out_bit_width-1 downto 0);

	-- Testbench signals
	signal dummy_addr		: std_logic_vector(0 downto 0);
	signal weights_rden		: std_logic; 
	signal thresholds_rden		: std_logic; 
	signal write_out		: std_logic; 

		
	component load_file is

		generic(
			word_length		: integer := 36;
			bram_addr_length	: integer := 6;
			addr_length		: integer := 16;
			N_bram			: integer := 58;
			N_words			: integer := 784;
			weights_filename	: string := "/home/alessio/"&
			"OneDrive/Dottorato/Progetti/SNN/spiker/vhdl/mark3/"&
			"hyperparameters/weights.mem"
		);

		port(
			-- input
			clk			: in std_logic;
			rden			: in std_logic;

			-- output
			di			: out std_logic_vector(
							word_length-1 downto 0);
			bram_addr		: out std_logic_vector(
							bram_addr_length-1
							downto 0);
			wraddr			: out std_logic_vector(
							addr_length-1 downto 0);
			wren			: out std_logic
		);

	end component load_file;


	component spiker is

		generic(
			-- Bit-widths
			neuron_bit_width	: integer := 16;
			weights_bit_width	: integer := 5;
			bram_word_length	: integer := 36;
			bram_addr_length	: integer := 10;
			bram_sel_length		: integer := 6;
			bram_we_length		: integer := 4;
			input_data_bit_width	: integer := 8;
			lfsr_bit_width		: integer := 16;
			cnt_out_bit_width	: integer := 16;

			-- Network shape
			inputs_addr_bit_width	: integer := 10;
			neurons_addr_bit_width	: integer := 9;

			-- Must be 1 bit longer than what required to count to N_cycles
			cycles_cnt_bit_width	: integer := 12;

			-- Bram parameters
			N_bram			: integer := 58;
			N_weights_per_word	: integer := 7;

			-- Structure parameters
			N_inputs		: integer := 784;
			N_neurons		: integer := 400;

			-- Internal parameters
			shift			: integer := 10
		);

		port(

			-- Common signals ---------------------------------------------
			clk			: in std_logic;
			rst_n			: in std_logic;	

			-- Input interface signals ------------------------------------
			init_lfsr		: in std_logic;
			seed			: in unsigned(lfsr_bit_width-1 
							downto 0);

			-- Layer signals ----------------------------------------------  

			-- control input
			start			: in std_logic;	
			init_v_th		: in std_logic;

			-- address to select the neurons
			v_th_addr		: in std_logic_vector(
							neurons_addr_bit_width
							  downto 0);

			-- data input
			input_data		: in unsigned(N_inputs*
							lfsr_bit_width-1 
							downto 0);

			-- input parameters
			v_th_value		: in signed(neuron_bit_width-1 
							downto 0);		
			v_reset			: in signed(neuron_bit_width-1 
							downto 0);	
			inh_weight		: in signed(neuron_bit_width-1
							downto 0);		

			-- terminal counters 
			N_inputs_tc		: in std_logic_vector (
							inputs_addr_bit_width 
							downto 0);
			N_neurons_tc		: in std_logic_vector(
							neurons_addr_bit_width 
							downto 0);
			N_cycles_tc		: in std_logic_vector(
							cycles_cnt_bit_width-1
							downto 0);

			-- output
			ready			: out std_logic;
			cnt_out			: out std_logic_vector(N_neurons*
							cnt_out_bit_width-1
							downto 0);


			-- Memory signals --------------------------------------------- 
			-- input
			di		: in std_logic_vector(bram_word_length-1 
						downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(bram_addr_length-1 
						downto 0);
			bram_sel	: in std_logic_vector(bram_sel_length-1 
						downto 0)
		);

	end component spiker;

begin

	v_reset		<= to_signed(v_reset_int, v_reset'length);
	inh_weight	<= to_signed(inh_weight_int, inh_weight'length);
	seed		<= to_unsigned(seed_int, seed'length);

	N_inputs_tc	<= std_logic_vector(to_signed(N_inputs,
			       N_inputs_tc'length));
	N_neurons_tc	<= std_logic_vector(to_signed(N_neurons,
				N_neurons_tc'length));
	N_cycles_tc	<= std_logic_vector(to_signed(N_cycles,
			       N_cycles_tc'length));

	v_th_addr(neurons_addr_bit_width-1) <= '0';
	dummy_addr	<= "0";

	-- clock
	clk_gen		: process
	begin
		clk <= '0';
		wait for 10 ns;
		clk <= '1';
		wait for 10 ns;
	end process clk_gen;

	-- reset (active low)
	rst_n_gen	: process
	begin
		rst_n <= '1';
		wait for 42 ns;
		rst_n <= '0';
		wait for 10 ns;
		rst_n <= '1';
		wait;
	end process rst_n_gen;

	-- weights read enable
	weights_rden_gen	: process
	begin
		weights_rden <= '0';
		wait for 100 ns;
		weights_rden <= '1';
		wait for 1 ms;
		weights_rden <= '0';
		wait;
	end process weights_rden_gen;

	-- thresholds read enable
	thresholds_rden_gen	: process
	begin
		thresholds_rden <= '0';
		wait for 1.1 ms;
		thresholds_rden <= '1';
		wait for 10 us;
		thresholds_rden <= '0';
		wait;
	end process thresholds_rden_gen;

	-- initialize lfsr
	init_lfsr_gen	: process
	begin
		init_lfsr <= '0';
		wait for 100 ns;
		init_lfsr <= '1';
		wait for 20 ns;
		init_lfsr <= '0';
		wait;
	end process init_lfsr_gen;


	-- read enable
	rden_gen	: process
	begin
		rden <= '0';
		wait for 1.2 ms;
		rden <= '1';
		wait;
	end process rden_gen;

	-- start generation
	start_gen	: process
	begin
		start <= '0';
		wait for 1.3 ms;
		start <= '1';
		wait for 20 ns;
		start <= '0';
		wait;
	end process start_gen;


	-- enable output write on file
	write_out_gen	: process
	begin
		write_out <= '0';
		wait for 10 ms;
		write_out <= '1';
		wait for 20 ns;
		write_out <= '0';
		wait;
	end process write_out_gen;


	-- quit simulation when the execution ends
	quit_sim	: process

		file output_file	: text open write_mode is
			output_filename;

		variable write_line	: line;

	begin

		wait until start = '1';
		wait until ready = '1';
		
		write(write_line, cnt_out);
		writeline(output_file, write_line);

		finish;

	end process quit_sim;




	-- initialize weights
	init_weights	: load_file 

		generic map(
			word_length		=> bram_word_length,
			bram_addr_length	=> bram_sel_length,
			addr_length		=> bram_addr_length,
			N_bram			=> N_bram,
			N_words			=> N_inputs,
			weights_filename	=> weights_filename
		)

		port map(
			-- input
			clk			=> clk,
			rden			=> weights_rden,

			-- output
			di			=> di,
			bram_addr		=> bram_sel,
			wraddr			=> wraddr,
			wren			=> wren
		);




	-- initialize thresholds
	init_thresholds : load_file 

		generic map(
			word_length		=> neuron_bit_width,
			bram_addr_length	=> 1,
			addr_length		=> neurons_addr_bit_width-1,
			N_bram			=> 1,
			N_words			=> N_neurons,
			weights_filename	=> thresholds_filename
		)

		port map(
			-- input
			clk			=> clk,
			rden			=> thresholds_rden,

			-- output
			std_logic_vector(di)	=> v_th_value,
			bram_addr		=> dummy_addr,
			wraddr			=> v_th_addr(neurons_addr_bit_width-2
							downto 0),
			wren			=> init_v_th 
		);





	-- read inputs from file
	read_inputs	: process

		file inputs_file	: text open read_mode is
			inputs_filename;

		variable read_line	: line;
		variable inputs_var	: std_logic_vector(N_inputs*
						input_data_bit_width-1 
						downto 0);

	begin

		-- Read line from file
		readline(inputs_file, read_line);
		read(read_line, inputs_var);

		-- Associate line to data input
		input_data	<= unsigned(inputs_var);
		wait;

	end process read_inputs;



	-- Store outputs on file
	-- store_outputs	: process(clk, ready)

	-- 	file output_file	: text open write_mode is
	-- 		output_filename;

	-- 	variable write_line	: line;

	-- begin

	-- 	if clk'event and clk = '1'
	-- 	then
	-- 		if write_out = '1'
	-- 		then

	-- 			write(write_line, cnt_out);
	-- 			writeline(output_file, write_line);

	-- 		end if;
	-- 	end if;	

	-- end process store_outputs;






	dut	: spiker
		generic map(
			-- Bit-widths
			neuron_bit_width	=> neuron_bit_width,
			weights_bit_width	=> weights_bit_width,
			bram_word_length	=> bram_word_length,
			bram_addr_length	=> bram_addr_length,
			bram_sel_length		=> bram_sel_length,
			bram_we_length		=> bram_we_length,
			input_data_bit_width	=> input_data_bit_width,
			lfsr_bit_width		=> lfsr_bit_width,
			cnt_out_bit_width	=> cnt_out_bit_width,
						   			
			-- Network shape
			inputs_addr_bit_width	=> inputs_addr_bit_width,
			neurons_addr_bit_width	=> neurons_addr_bit_width,
						   			
			-- Must be 1 bit longer
			cycles_cnt_bit_width	=> cycles_cnt_bit_width,
						   			
			-- Bram parameters
			N_bram			=> N_bram,
			N_weights_per_word	=> N_weights_per_word,
						   			
			-- Structure parameters
			N_inputs		=> N_inputs,
			N_neurons		=> N_neurons,
						   			
			-- Internal parameters
			shift			=> shift			
		)

		port map(

			-- Inputs
			clk			=> clk,
			rst_n			=> rst_n,
			init_lfsr		=> init_lfsr,
			seed			=> seed,
			start			=> start,
			init_v_th		=> init_v_th,
			v_th_addr		=> v_th_addr,
			input_data		=> input_data,
			v_th_value		=> v_th_value,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			N_inputs_tc		=> N_inputs_tc,
			N_neurons_tc		=> N_neurons_tc,
			N_cycles_tc		=> N_cycles_tc,
			di			=> di,
			rden			=> rden,
			wren			=> wren,
			wraddr			=> wraddr,
			bram_sel		=> bram_sel,

			-- Outputs
			ready			=> ready,
			cnt_out			=> cnt_out
		);

end architecture test;
