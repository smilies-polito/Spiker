library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity weights_bram_tb is
end entity weights_bram_tb;

architecture test of weights_bram_tb is

	constant weights_filename	: string	:= "/home/alessio/Documents/"&
		"Poli/Dottorato/progetti/snn/spiker/vhdl/mark3/"&
		"hyperparameters/weights1.mem";

	constant out_filename	: string	:= "out_file.mem";

	-- input
	signal clk		: std_logic;
	signal di		: std_logic_vector(35 downto 0);
	signal rst_n		: std_logic;
	signal rdaddr		: std_logic_vector(9 downto 0);
	signal rden		: std_logic;
	signal wren		: std_logic;
	signal wraddr		: std_logic_vector(9 downto 0);
	signal bram_sel		: std_logic_vector(5 downto 0);

	-- output
	signal do		: std_logic_vector(400*5-1 downto 0);

	signal rd_ok		: std_logic;

	
	component weights_bram is

		port(
			-- input
			clk		: in std_logic;
			di		: in std_logic_vector(35 downto 0);
			rst_n		: in std_logic;
			rdaddr		: in std_logic_vector(9 downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(9 downto 0);
			bram_sel	: in std_logic_vector(5 downto 0);

			-- output
			do		: out std_logic_vector(400*5-1 downto 0)
					
		);

	end component weights_bram;

begin

	
	bram_sel	<= std_logic_vector(to_unsigned(0, bram_sel'length));
	rst_n		<= '1';


	-- clock
	clock_gen : process
	begin
		clk	<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- write data inside the bram
	write_process : process(clk)

		file weights_file	: text open read_mode is
						weights_filename;

		variable read_line	: line;
		variable di_var		: std_logic_vector(35 downto 0);
		variable addr_var	: integer := 0;

	begin

		if clk'event and clk = '1'
		then

			if not endfile(weights_file)
			then

				readline(weights_file, read_line);
				read(read_line, di_var);

				di	<= di_var;
				wraddr	<= std_logic_vector(
						to_unsigned(addr_var,
							wraddr'length));
				wren	<= '1';

				addr_var := addr_var + 1;
				

			else

				wren	<= '0';

			end if;


		end if;

	end process write_process;



	-- read data from the bram
	read_process : process(clk)

		file out_file		: text open write_mode is
						out_filename;

		variable write_line	: line;
		variable do_var		: std_logic_vector(35 downto 0);
		variable addr_var	: integer := 0;

	begin

		if clk'event and clk = '1'
		then

			if rd_ok = '1'
			then

				rdaddr	<= std_logic_vector(
						to_unsigned(addr_var,
							wraddr'length));
				rden	<= '1';

				do_var := '0' & do(34 downto 0);

				write(write_line, do_var);
				writeline(out_file, write_line);

				addr_var := addr_var + 1;
				

			else

				rden	<= '0';

			end if;


		end if;

	end process read_process;


	-- enable read process
	rd_ok_gen	: process
	begin

		rd_ok <= '0';
		wait for 12000 ns; 
		rd_ok <= '1';
		wait for 9408 ns; 
		rd_ok <= '1';
		wait;

	end process rd_ok_gen;


	-- read data from the bram
	-- read_process	: process(clk, rd_ok)

	-- 	variable addr_var	: integer := 0;

	-- begin

	-- 	if clk'event and clk = '1'
	-- 	then
	-- 		if rd_ok = '1'
	-- 		then
	-- 			rden 	<= '1';
	-- 			rdaddr	<= std_logic_vector(
	-- 					to_unsigned(addr_var, 10));
	-- 		addr_var := addr_var + 1;
	-- 		else
	-- 			rden	<= '0';
	-- 		end if;

	-- 	end if;

	-- end process read_process;


	dut	: weights_bram 

		port map(
			-- input
			clk		=> clk,
			di		=> di,
			rst_n		=> rst_n,
			rdaddr		=> rdaddr,
			rden		=> rden,
			wren		=> wren,
			wraddr		=> wraddr,
			bram_sel	=> bram_sel,

			-- output
			do		=> do
					
		);


end architecture test;
