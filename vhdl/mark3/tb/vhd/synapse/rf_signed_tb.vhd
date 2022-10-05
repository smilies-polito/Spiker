library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity rf_signed_tb is
end entity rf_signed_tb;

architecture test of rf_signed_tb is


	constant word_length		: integer := 10;
	constant N_weights_per_word	: integer := 2;
	constant rdwr_addr_length	: integer := 2;
	constant N_neurons		: integer := 2;
	constant weights_bit_width	: integer := 5;
	constant N_bram			: integer := 1;

	-- input
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

	-- testbench signals
	signal rdwr		: std_logic_vector(1 downto 0);

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

	-- clk
	clk_gen	: process
	begin

		clk	<= '0';
		wait for 10 ns;
		clk	<= '1';
		wait for 10 ns;

	end process clk_gen;

	-- rdwr
	rdwr_gen	: process
	begin

		rdwr	<= "00";
		wait for 25 ns;
		rdwr	<= "11";
		wait for 60 ns;
		rdwr	<= "00";
		wait for 20 ns;
		rdwr	<= "01";
		wait for 20 ns;
		rdwr	<= "10";
		wait for 200 ns;
		rdwr	<= "01";
		wait for 20 ns;
		rdwr	<= "10";
		wait for 200 ns;
		rdwr	<= "00";
		wait;

	end process rdwr_gen;

	-- write and then read the rf
	read_write	: process(clk, rdwr)

		variable wraddr_var	: integer := 0;
		variable rdaddr_var	: integer := 0;

	begin

		if clk'event and clk = '1'
		then

			wren	<= '0';
			wraddr	<= (others => '0');
			rden	<= '0';
			rdaddr	<= (others => '0');
			rst_n	<= '1';

			if rdwr = "11"
			then

				di <= std_logic_vector(to_signed(wraddr_var,
				      weights_bit_width)) & std_logic_vector(
				      to_signed(wraddr_var + 5, weights_bit_width));
				wren <= '1';
				wraddr <= std_logic_vector(to_unsigned(wraddr_var,
					  rdwr_addr_length));

				wraddr_var := wraddr_var + 1;

			elsif rdwr = "00"
			then
				
				rst_n <= '0';

			elsif rdwr = "01"
			then
				
				rden	<= '0';
				
			elsif rdwr = "10"
			then

				rden <= '1';
				rdaddr <= std_logic_vector(to_unsigned(rdaddr_var,
					  rdwr_addr_length));

				rdaddr_var := rdaddr_var + 1;

			end if;

		end if;

	end process read_write;


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
