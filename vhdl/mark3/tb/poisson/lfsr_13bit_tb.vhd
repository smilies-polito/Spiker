library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;


entity lfsr_13bit_tb is
end entity lfsr_13bit_tb;



architecture test of lfsr_13bit_tb is

	-- input
	signal clk		: std_logic;
	signal en		: std_logic;
	signal load_n		: std_logic;
	signal lfsr_in		: std_logic_vector(12 downto 0);

	-- output
	signal lfsr_out		: std_logic_vector(12 downto 0);


	component lfsr_13bit is

		port(
			-- input
			clk		: in std_logic;
			en		: in std_logic;
			load_n		: in std_logic;
			lfsr_in		: in std_logic_vector(12 downto 0);

			-- output
			lfsr_out	: out std_logic_vector(12 downto 0)
		);

	end component lfsr_13bit;


begin

	file_write	: process(clk)

		file lfsr_file	: text open write_mode is
			"/home/alessio/Documents/Poli/Dottorato/progetti/snn" &
			"/spiker/vhdl/mark2/sim/files/lfsr_out.txt";

		variable row		: line;
		variable write_var	: integer;

	begin

		write_var	:= to_integer(unsigned(lfsr_out));
		
		if clk'event and clk = '1'
		then

			if en = '1'
			then
				write(row, write_var);
				writeline(lfsr_file, row);
			end if;
		end if;

	end process file_write;


	lfsr_in	<= std_logic_vector(to_unsigned(186, 13)); 
	
	-- clock
	clock_gen 	: process
	begin
		clk		<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk		<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- enable
	en_gen	: process
	begin
		en		<= '1';		-- 0 ns
		wait for 14 ns;
		en		<= '0';		-- 14 ns
		wait for 48 ns;
		en 		<= '1';		-- 62 ns
		wait;
	end process en_gen;


	-- load_n
	load_n_gen	: process
	begin
		load_n	<= '0';		-- 0 ns
		wait for 14 ns;
		load_n 	<= '1';		-- 14 ns
		wait;
	end process load_n_gen;




	dut	: lfsr_13bit 

		port map(
			-- input
			clk		=> clk,
			en		=> en,
			load_n		=> load_n,
			lfsr_in		=> lfsr_in,

			-- output
			lfsr_out	=> lfsr_out
		);


end architecture test;
