library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity rf_signed_tb is
end entity rf_signed_tb;

architecture test of rf_signed_tb is


	constant word_length		: integer := 36;
	constant N_weights_per_word	: integer := 2;
	constant rdwr_addr_length	: integer := 10;
	constant N_neurons		: integer := 400;
	constant weights_bit_width	: integer := 5;
	constant N_bram			: integer := 58;


	signal clk		: std_logic;
	signal di		: std_logic_vector(word_length-1 downto 0);
	signal rst_n		: std_logic;
	signal rdaddr		: std_logic_vector(rdwr_addr_length-1 
					downto 0);
	signal rden		: std_logic;
	signal wren		: std_logic;
	signal wraddr		: std_logic_vector(rdwr_addr_length-1
					downto 0);

	-- output
	signal do		: std_logic_vector(N_weights_per_word*
				weights_bit_width-1 downto 0);

	component rf_signed is

		generic(
			word_length		: integer := 36;
			N_weights_per_word	: integer := 2;
			rdwr_addr_length	: integer := 10;
			N_neurons		: integer := 400;
			weights_bit_width	: integer := 5;
			N_bram			: integer := 58
		);

		port(
			-- input
			clk		: in std_logic;
			di		: in std_logic_vector(word_length-1
						downto 0);
			rst_n		: in std_logic;
			rdaddr		: in std_logic_vector(rdwr_addr_length-1 
						downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(rdwr_addr_length-1
						downto 0);

			-- output
			do		: out std_logic_vector(
						N_weights_per_word*
						weights_bit_width-1 downto 0)
					
		);

	end component rf_signed;

begin

	dut	: rf_signed 

		generic map(
			word_length		=> word_length,
			N_weights_per_word	=> N_weights_per_word,
			rdwr_addr_length	=> rdwr_addr_length,
			N_neurons		=> N_neurons,
			weights_bit_width	=> weights_bit_width,
			N_bram			=> N_bram
		)                                                          
                                                                           
		port map(
			-- input
			clk			=> clk,
			di			=> di,
			rst_n			=> rst_n,
			rdaddr			=> rdaddr,
			rden			=> rden,
			wren			=> wren,
			wraddr			=> wraddr,

			-- output
			do			=> do
					
		);

end architecture test;
