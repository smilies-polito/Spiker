library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity load_file_tb is
end entity load_file_tb;

architecture behaviour of load_file_tb is


	constant word_length		: integer := 4;
	constant bram_addr_length	: integer := 2;
	constant addr_length		: integer := 2;
	constant N_bram			: integer := 4;
	constant N_words		: integer := 4;
	constant weights_filename	: string := "/home/alessio/"&
		"OneDrive/Dottorato/Progetti/SNN/spiker/vhdl/mark3/"&
		"sim/prova.mem";


	-- input
	signal clk			: std_logic;
	signal rden			: std_logic;

	-- output
	signal di			: std_logic_vector(word_length-1
					    downto 0);
	signal bram_addr		: std_logic_vector(bram_addr_length
					    -1 downto 0);
	signal wraddr			: std_logic_vector(addr_length-1
					    downto 0);
	signal wren			: std_logic;

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
			di			: out std_logic_vector(word_length-1
							downto 0);
			bram_addr		: out std_logic_vector(bram_addr_length
							-1 downto 0);
			wraddr			: out std_logic_vector(addr_length-1
							downto 0);
			wren			: out std_logic
		);

	end component load_file;

begin

	-- clock
	clk_gen		: process
	begin
		clk <= '0';
		wait for 10 ns;
		clk <= '1';
		wait for 10 ns;
	end process clk_gen;

	-- rden
	rden_gen	: process
	begin
		rden <= '0';
		wait for 42 ns;
		rden <= '1';
		wait;
	end process rden_gen;




	dut	: load_file 

		generic map(
			word_length		=> word_length,
			bram_addr_length	=> bram_addr_length,
			addr_length		=> addr_length,
			N_bram			=> N_bram,
			N_words			=> N_words,
			weights_filename	=> weights_filename
		)

		port map(
			-- input
			clk			=> clk,
			rden			=> rden,

			-- output
			di			=> di,
			bram_addr		=> bram_addr,
			wraddr			=> wraddr,
			wren			=> wren
		);

end architecture behaviour;
