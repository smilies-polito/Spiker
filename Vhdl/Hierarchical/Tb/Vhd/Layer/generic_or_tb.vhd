library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity generic_or_tb is
end entity generic_or_tb;


architecture test of generic_or_tb is

	constant N	: integer := 4;

	signal or_in	: std_logic_vector(N-1 downto 0);
	signal or_out	: std_logic;


	component generic_or is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			or_in	: in std_logic_vector(N-1 downto 0);

			-- output
			or_out	: out std_logic
		);

	end component generic_or;



begin


	or_in_gen	: process
	begin

		for i in 0 to 2**N-1
		loop

			or_in	<= std_logic_vector(to_unsigned(i, N));
			wait for 10 ns;

		end loop;

		wait;

	end process or_in_gen;




	dut	: generic_or 

		generic map(
			N	=> N	
		)

		port map(
			-- input
			or_in	=> or_in,	

			-- outpu=>t
			or_out	=> or_out	
		);


end architecture test;
