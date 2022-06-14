library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;


entity lfsr_tb is
end entity lfsr_tb;



architecture test of lfsr_tb is

	constant bit_width	: integer := 16;
	constant seed_int	: integer := 5;

	-- input
	signal clk		: std_logic;
	signal update		: std_logic;
	signal load		: std_logic;
	signal seed		: unsigned(bit_width-1 downto 0);

	-- output
	signal pseudo_random	: unsigned(bit_width-1 downto 0);



	component lfsr is

		generic(
			bit_width	: integer := 16
		);

		port(
			-- control input
			clk		: in std_logic;
			load		: in std_logic;
			update		: in std_logic;

			-- data input
			seed		: in unsigned(bit_width-1 downto 0);

			-- output
			pseudo_random	: out unsigned(bit_width-1 downto 0)
		);

	end component lfsr;

begin

	file_write	: process(clk)

		file lfsr_file	: text open write_mode is
			"/home/alessio/Documents/Poli/Dottorato/Progetti/"&
			"spiker/vhdl/mark3/sim/inputOutput/lfsr_vhd_out.txt";
		variable row		: line;
		variable write_var	: integer;

	begin

		write_var	:= to_integer(unsigned(pseudo_random));
		
		if clk'event and clk = '1'
		then

			if update = '1'
			then
				write(row, write_var);
				writeline(lfsr_file, row);
			end if;
		end if;

	end process file_write;


	seed	<= to_unsigned(seed_int, bit_width); 
	
	-- clock
	clock_gen 	: process
	begin
		clk		<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk		<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- update lfsr
	update_gen	: process
	begin
		update		<= '0';		-- 14 ns
		wait for 48 ns;
		update 		<= '1';		-- 62 ns
		wait;
	end process update_gen;


	-- load
	load_gen	: process
	begin
		load	<= '1';		-- 0 ns
		wait for 14 ns;
		load 	<= '0';		-- 14 ns
		wait;
	end process load_gen;




	dut	: lfsr

		generic map(
			bit_width	=> bit_width
		)

		port map(
			-- control input
			clk		=> clk,
			load		=> load,
			update		=> update,

			-- data input
			seed		=> seed,

			-- output
			pseudo_random	=> pseudo_random
		);


end architecture test;
