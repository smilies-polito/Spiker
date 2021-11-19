library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity generic_and_tb is
end entity generic_and_tb;


architecture test of generic_and_tb is

	constant N	: integer := 4;

	signal and_in	: std_logic_vector(N-1 downto 0);
	signal and_out	: std_logic;


	component generic_and is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			and_in	: in std_logic_vector(N-1 downto 0);

			-- output
			and_out	: out std_logic
		);

	end component generic_and;



begin


	and_in_gen	: process
	begin

		for i in 0 to 2**N-1
		loop

			and_in	<= std_logic_vector(to_unsigned(i, N));
			wait for 10 ns;

		end loop;

		wait;

	end process and_in_gen;




	dut	: generic_and 

		generic map(
			N	=> N	
		)

		port map(
			-- input
			and_in	=> and_in,	

			-- outpu=>t
			and_out	=> and_out	
		);


end architecture test;
