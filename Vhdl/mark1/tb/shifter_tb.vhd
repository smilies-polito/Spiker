library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity shifter_tb is
end entity shifter_tb;


architecture test of shifter_tb is


	constant N		: integer := 8;
	constant shift		: integer := 3;

	signal shifter_in	: signed(N-1 downto 0);
	signal shifted_out	: signed(N-1 downto 0);


	component shifter is

		generic(
			-- parallelism
			N		: integer := 8;
		
			-- shift
			shift		: integer := 1	
		);

		port(
			-- input
			shifter_in	: in signed(N-1 downto 0);

			-- output
			shifted_out	: out signed(N-1 downto 0)
		);

	end component shifter;


begin



	shift_test 	: process
	begin

	shifter_in	<= "01000000";

	wait for 10 ns;

	shifter_in	<= "01001000";

	wait;
	

	end process shift_test;



	dut	: shifter
		generic map(
			N		=> N,
			shift		=> shift		
		)
		port map(
			shifter_in	=> shifter_in,
			shifted_out	=> shifted_out
		);

end architecture test;
