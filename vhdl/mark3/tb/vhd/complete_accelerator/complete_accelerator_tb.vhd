library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity complete_accelerator_tb is
end entity complete_accelerator_tb;

-- 3 inputs, 2 neurons, 30 cycles
architecture test_3x2x30 of complete_accelerator_tb is

	-- Network parameters
	constant v_reset_int		: integer := 5*2**3; 	  
	constant inh_weight_int	 	: integer := -15*2**3; 
	constant seed_int		: integer := 5;
	constant N_inputs		: integer := 3;
	constant N_neurons		: integer := 2;
	constant N_cycles		: integer := 30;

	-- Hyperparameters
	constant v_th_0_int		: integer := 13*2**3;
	constant v_th_1_int		: integer := 8*2**3;
	-- constant v_th_1_int		: integer := 13*2**3;
	constant weight_0_0_int		: integer := 15*2**3;
	-- constant weight_0_1_int		: integer := 5*2**3;
	-- constant weight_0_2_int		: integer := 5*2**3;
	constant weight_0_1_int		: integer := 9*2**3;
	constant weight_0_2_int		: integer := 5*2**3;
	-- constant weight_1_0_int		: integer := 3*2**3;
	-- constant weight_1_1_int		: integer := 3*2**3;
	-- constant weight_1_2_int		: integer := 3*2**3;
	-- constant weight_1_1_int		: integer := 9*2**3;
	-- constant weight_1_2_int		: integer := 5*2**3;
	constant weight_1_0_int		: integer := 5*2**3;
	constant weight_1_1_int		: integer := 3*2**3;
	constant weight_1_2_int		: integer := 9*2**3;

	constant pixel0			: integer := 2**8-1;
	constant pixel1			: integer := 2**8-1;
	constant pixel2			: integer := 2**8-1;
	
	-- Interface bit-widths
	constant word_length		: integer := 64;
	constant load_bit_width		: integer := 4;
	constant data_bit_width		: integer := 36;
	constant addr_bit_width		: integer := 10;
	constant sel_bit_width		: integer := 10;

	-- Internal bit-widths
	constant neuron_bit_width	: integer := 16;
	constant weights_bit_width	: integer := 15;
	constant bram_word_length	: integer := 36;
	constant bram_addr_length	: integer := 10;
	constant bram_sel_length	: integer := 1;
	constant bram_we_length		: integer := 4;
	constant input_data_bit_width	: integer := 8;
	constant lfsr_bit_width		: integer := 16;
	constant cnt_out_bit_width	: integer := 16;

	-- Network shape
	constant inputs_addr_bit_width	: integer := 2;
	constant neurons_addr_bit_width	: integer := 1;

	-- Must be 1 bit longer than what required to count to
	-- N_cycles
	constant cycles_cnt_bit_width	: integer := 6;

	-- Bram parameters
	constant N_bram			: integer := 1;
	constant N_weights_per_word	: integer := 2;

	-- Internal parameters
	constant shift			: integer := 10;

	-- Initialization files
	constant weights_filename	: string  := "/home/alessio/"&
		"Documents/Poli/Dottorato/Progetti/spiker/vhdl/mark3/"&
		"tb/vhd/complete_accelerator/weights.txt";
	constant v_th_filename		: string  := "/home/alessio/"&
		"Documents/Poli/Dottorato/Progetti/spiker/vhdl/mark3/"&
		"tb/vhd/complete_accelerator/v_th.txt";
	constant inputs_filename	: string  := "/home/alessio/"&
		"Documents/Poli/Dottorato/Progetti/spiker/vhdl/mark3/"&
		"tb/vhd/complete_accelerator/pixels.txt";
	constant cnt_out_filename	: string  := "/home/alessio/"&
		"Documents/Poli/Dottorato/Progetti/spiker/vhdl/mark3/"&
		"tb/vhd/complete_accelerator/cnt_out.txt";



	-- Driver signals
	signal driver_rst_n		: std_logic;
	signal go			: std_logic;
	signal N_inputs_cnt		: std_logic_vector(addr_bit_width-1
					     downto 0);
	signal N_neurons_cnt		: std_logic_vector(addr_bit_width-1
					     downto 0);
	signal input_word		: std_logic_vector(word_length-1 
						downto 0);

	-- Output
	signal N_neurons_cnt_en		: std_logic;
	signal N_inputs_cnt_en		: std_logic;
	signal N_neurons_cnt_rst_n	: std_logic;
	signal N_inputs_cnt_rst_n	: std_logic;
	signal output_word		: std_logic_vector(word_length-1
						downto 0);
	

	-- Input
	signal clk			: std_logic;
	signal rst_n			: std_logic;	
	signal start			: std_logic;	
	signal addr			: std_logic_vector(
					     addr_bit_width-1
					     downto 0);
	signal load			: std_logic_vector(
					     load_bit_width-1
					     downto 0);
	signal data			: std_logic_vector(
					     data_bit_width-1 
					     downto 0);
	signal sel			: std_logic_vector(
						sel_bit_width-1
						downto 0);
	-- Output
	signal ready			: std_logic;
	signal cnt_out			: std_logic_vector(
						cnt_out_bit_width-1
						downto 0);

	-- Memory signals --------------------------------------
	signal rden			: std_logic;


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
			cnt_out		: out std_logic_vector(bit_width-1 downto 0)		
		);

	end component cnt;


	component driver is
		generic(
			-- Bit-width
			word_length		: integer := 64;
			load_bit_width		: integer := 4;
			data_bit_width		: integer := 36;
			addr_bit_width		: integer := 10;
			sel_bit_width		: integer := 10;

			-- Internal parameters
			N_inputs_tc_value	: integer := 3;
			N_neurons_tc_value	: integer := 2;
			N_cycles_tc_value	: integer := 30;
			v_reset_value		: integer := 5;
			inh_weight_value	: integer := -15;
			seed_value		: integer := 5;

			-- Initialization files
			weights_filename	: string  := "";
			v_th_filename		: string  := "";
			inputs_filename		: string  := "";
			cnt_out_filename	: string  := ""
				
		);
		port(

			-- input
			clk			: in std_logic;
			driver_rst_n		: in std_logic;
			go			: in std_logic;
			N_inputs_cnt		: in std_logic_vector(
							addr_bit_width-1
							downto 0);
			N_neurons_cnt		: in std_logic_vector(
							addr_bit_width-1
							downto 0);
			input_word		: in std_logic_vector(word_length-1 
							downto 0);

			-- output
			N_neurons_cnt_en	: out std_logic;
			N_inputs_cnt_en		: out std_logic;
			N_neurons_cnt_rst_n	: out std_logic;
			N_inputs_cnt_rst_n	: out std_logic;
			output_word		: out std_logic_vector(word_length-1
							downto 0)
		    
		);
	end component driver;



	component complete_accelerator is

		generic(

			-- Interface bit-widths
			load_bit_width		: integer := 4;
			data_bit_width		: integer := 36;
			addr_bit_width		: integer := 10;
			sel_bit_width		: integer := 10;

			-- Internal bit-widths
			neuron_bit_width	: integer := 16;
			weights_bit_width	: integer := 15;
			bram_word_length	: integer := 36;
			bram_addr_length	: integer := 10;
			bram_sel_length		: integer := 1;
			bram_we_length		: integer := 4;
			input_data_bit_width	: integer := 8;
			lfsr_bit_width		: integer := 16;
			cnt_out_bit_width	: integer := 16;

			-- Network shape
			inputs_addr_bit_width	: integer := 2;
			neurons_addr_bit_width	: integer := 1;

			-- Must be 1 bit longer than what required to count to
			-- N_cycles
			cycles_cnt_bit_width	: integer := 6;

			-- Bram parameters
			N_bram			: integer := 1;
			N_weights_per_word	: integer := 2;

			-- Structure parameters
			N_inputs		: integer := 3;
			N_neurons		: integer := 2;

			-- Internal parameters
			shift			: integer := 10

		);

		port(

			-- Input
			clk			: in std_logic;
			rst_n			: in std_logic;	
			start			: in std_logic;	
			addr			: in std_logic_vector(
							addr_bit_width-1
							downto 0);
			load			: in std_logic_vector(
							load_bit_width-1
							downto 0);
			data			: in std_logic_vector(
							data_bit_width-1 
							downto 0);
			sel			: in std_logic_vector(
							sel_bit_width-1
							downto 0);
			-- Output
			ready			: out std_logic;
			cnt_out			: out std_logic_vector(
							cnt_out_bit_width-1
							downto 0);

			-- Memory signals --------------------------------------
			rden			: in std_logic
		);

	end component complete_accelerator;

begin

	store_init_values	: process

		file weights_file	: text open write_mode is
						weights_filename;

		file v_th_file		: text open write_mode is
						v_th_filename;

		file inputs_file	: text open write_mode is
						inputs_filename;

		variable write_line	: line;

		variable data_var	: std_logic_vector(data'length-1 downto
						0);

	begin

		-- Threshold 0
		data_var := std_logic_vector(to_unsigned(v_th_0_int,
				data_var'length));
		write(write_line, data_var);
		writeline(v_th_file, write_line);

		-- Threshold 1
		data_var := std_logic_vector(to_unsigned(v_th_1_int,
				data_var'length));
		write(write_line, data_var);
		writeline(v_th_file, write_line);

		-- Weights 00 and 10
		data_var(weights_bit_width-1 downto 0) :=
			std_logic_vector(to_unsigned(weight_0_0_int,
			weights_bit_width));
		data_var(2*weights_bit_width-1 downto weights_bit_width) :=
			std_logic_vector(to_unsigned(weight_1_0_int,
			weights_bit_width));
		data_var(data'length-1 downto 2*weights_bit_width) :=
			(others => '0');
		write(write_line, data_var);
		writeline(weights_file, write_line);

		-- Weights 01 and 11
		data_var(weights_bit_width-1 downto 0) :=
			std_logic_vector(to_unsigned(weight_0_1_int,
			weights_bit_width));
		data_var(2*weights_bit_width-1 downto weights_bit_width) :=
			std_logic_vector(to_unsigned(weight_1_1_int,
			weights_bit_width));
		data_var(data'length-1 downto 2*weights_bit_width) :=
			(others => '0');
		write(write_line, data_var);
		writeline(weights_file, write_line);

		-- Weights 02 and 12
		data_var(weights_bit_width-1 downto 0) :=
			std_logic_vector(to_unsigned(weight_0_2_int,
			weights_bit_width));
		data_var(2*weights_bit_width-1 downto weights_bit_width) :=
			std_logic_vector(to_unsigned(weight_1_2_int,
			weights_bit_width));
		data_var(data'length-1 downto 2*weights_bit_width) :=
			(others => '0');
		write(write_line, data_var);
		writeline(weights_file, write_line);

		-- Pixel 0
		data_var := std_logic_vector(to_unsigned(pixel0,
				data_var'length));
		write(write_line, data_var);
		writeline(inputs_file, write_line);
		
		-- Pixel 1
		data_var := std_logic_vector(to_unsigned(pixel1,
				data_var'length));
		write(write_line, data_var);
		writeline(inputs_file, write_line);

		-- Pixel 2
		data_var := std_logic_vector(to_unsigned(pixel2,
				data_var'length));
		write(write_line, data_var);
		writeline(inputs_file, write_line);

		wait;

	end process store_init_values;


	input_word(input_word'length-1)		<= ready;
	input_word(input_word'length-2 
		downto
		cnt_out_bit_width)		<= (others => '0');
	input_word(cnt_out_bit_width-1 
		downto 0)			<= cnt_out;


	data	<= output_word(
			data_bit_width-1 
			downto 
			0);

	addr	<= output_word(
			data_bit_width+
			addr_bit_width-1 
			downto 
			data_bit_width);

	sel	<= output_word(
			data_bit_width+
			addr_bit_width+
			sel_bit_width-1
			downto 
			data_bit_width+
			addr_bit_width);

	load	<= output_word(
			data_bit_width+
			addr_bit_width+
			sel_bit_width+
			load_bit_width-1
			downto 
			data_bit_width+
			addr_bit_width+
			sel_bit_width);

	start	<= output_word(
			data_bit_width+
			addr_bit_width+
			sel_bit_width+
			load_bit_width);

	rst_n	<= output_word(
			data_bit_width+
			addr_bit_width+
			sel_bit_width+
			load_bit_width+1);


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
		wait;
	end process go_gen;

	-- rden
	rden_gen	: process
	begin
		rden <= '0';
		wait for 20 ns;
		rden <= '1';
		wait;
	end process rden_gen;






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



	simulated_driver	: driver
		generic map(
			-- Bit-width
			word_length		=> word_length,
			load_bit_width		=> load_bit_width,
			data_bit_width		=> data_bit_width,
			addr_bit_width		=> addr_bit_width,
			sel_bit_width		=> sel_bit_width,

			-- Internal parameters
			N_inputs_tc_value	=> N_inputs - 1,
			N_neurons_tc_value	=> N_neurons - 1,
			N_cycles_tc_value	=> N_cycles - 1,
			v_reset_value		=> v_reset_int,	
			inh_weight_value	=> inh_weight_int,
			seed_value		=> seed_int,

			-- Initialization files
			weights_filename	=> weights_filename,
			v_th_filename		=> v_th_filename,
			inputs_filename		=> inputs_filename,
			cnt_out_filename	=> cnt_out_filename
		)
		port map(

			-- input
			clk			=> clk,
			driver_rst_n		=> driver_rst_n,
			go			=> go,
			N_inputs_cnt		=> N_inputs_cnt,
			N_neurons_cnt		=> N_neurons_cnt,
			input_word		=> input_word,

			-- output
			N_neurons_cnt_en	=> N_neurons_cnt_en,
			N_inputs_cnt_en		=> N_inputs_cnt_en,
			N_neurons_cnt_rst_n	=> N_neurons_cnt_rst_n,
			N_inputs_cnt_rst_n	=> N_inputs_cnt_rst_n,
			output_word		=> output_word
		);


	dut	: complete_accelerator 

		generic map(

			-- Interface bit-widths
			load_bit_width		=> load_bit_width,
			data_bit_width		=> data_bit_width,
			addr_bit_width		=> addr_bit_width,
			sel_bit_width		=> sel_bit_width,

			-- Internal bit-widths
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

			-- Must be 1 bit longer than what required to count to
			-- N_cycles
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

			-- Input
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,
			addr			=> addr,
			load			=> load,
			data			=> data,
			sel			=> sel,
			
			-- Output
			ready			=> ready,
			cnt_out			=> cnt_out,

			-- Memory signals --------------------------------------
			rden			=> rden	
		);

end architecture test_3x2x30;
