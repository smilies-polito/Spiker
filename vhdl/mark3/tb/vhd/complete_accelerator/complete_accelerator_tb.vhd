library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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
	constant weight_0_0_int		: integer := 15*2**3;
	constant weight_0_1_int		: integer := 9*2**3;
	constant weight_0_2_int		: integer := 5*2**3;
	constant weight_1_0_int		: integer := 5*2**3;
	constant weight_1_1_int		: integer := 3*2**3;
	constant weight_1_2_int		: integer := 9*2**3;
	
	-- Interface bit-widths
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

	-- clock
	clk_gen		: process
	begin
		clk <= '0';
		wait for 10 ns;
		clk <= '1';
		wait for 10 ns;
	end process clk_gen;



	interface_accelerator	: process
	begin

		-- Default values
		rst_n		<= '0';
		start		<= '0';
		sel		<= (others => '0');
		load		<= (others => '1');
		addr		<= (others => '0');
		data		<= (others => '0');


		wait;

	end process interface_accelerator;


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
