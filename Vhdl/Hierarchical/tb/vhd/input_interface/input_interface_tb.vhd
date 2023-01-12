library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity input_interface_tb is
end entity input_interface_tb;

architecture test of input_interface_tb is


	constant bit_width	: integer := 16;
	constant N_inputs	: integer := 784;
	constant seed_int	: integer := 5;

	-- control input
	signal clk		: std_logic;
	signal load		: std_logic;
	signal update		: std_logic;

	-- data input
	signal seed		: unsigned(bit_width-1 downto 0);
	signal input_data	: unsigned(N_inputs*bit_width-1 downto 0);

	-- output
	signal output_spikes	: std_logic_vector(N_inputs-1 downto 0);


	component input_interface is

		generic(
			bit_width	: integer := 16;
			N_inputs	: integer := 784
		);

		port(
			-- control input
			clk		: in std_logic;
			load		: in std_logic;
			update		: in std_logic;

			-- data input
			seed		: in unsigned(bit_width-1 downto 0);
			input_data	: in unsigned(N_inputs*bit_width-1 downto 0);

			-- output
			output_spikes	: out std_logic_vector(N_inputs-1 downto 0)
		);

	end component input_interface;

begin


	seed	<= to_unsigned(seed_int, bit_width); 


	-- Read the input image from file
	file_read	: process

		file input_file	: text open read_mode is
			"/home/alessio/Documents/Poli/Dottorato/Progetti/"&
			"spiker/vhdl/mark3/sim/inputOutput/vhdlImage.txt";
		variable read_line	: line;
		variable read_var	: std_logic_vector(N_inputs*bit_width-1 downto 0);

	begin

		-- Read line from file
		readline(input_file, read_line);
		read(read_line, read_var);

		-- Associate line to data input
		input_data	<= unsigned(read_var);
		wait;

	end process file_read;



	file_write	: process(clk)

		file lfsr_file	: text open write_mode is
			"/home/alessio/Documents/Poli/Dottorato/Progetti/"&
			"spiker/vhdl/mark3/sim/inputOutput/vhdlSpikes.txt";
		variable row		: line;
		variable write_var	: std_logic_vector(N_inputs-1 downto 0);

	begin

		write_var	:= output_spikes;
		
		if clk'event and clk = '1'
		then

			if update = '1'
			then
				write(row, write_var);
				writeline(lfsr_file, row);
			end if;
		end if;

	end process file_write;


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




	dut	: input_interface

		generic map(
			bit_width	=> bit_width,
			N_inputs	=> N_inputs
		)

		port map(
			-- control input
			clk		=> clk,
			load		=> load,
			update		=> update,

			-- data input
			seed		=> seed,
			input_data	=> input_data,

			-- output
			output_spikes	=> output_spikes
		);

end architecture test;
