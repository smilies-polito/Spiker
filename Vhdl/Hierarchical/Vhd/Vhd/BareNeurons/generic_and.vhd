library ieee;
use ieee.std_logic_1164.all;

entity generic_and is

	generic(
		N	: integer := 8		
	);

	port(
		-- input
		and_in	: in std_logic_vector(N-1 downto 0);

		-- output
		and_out	: out std_logic
	);

end entity generic_and;


architecture behaviour of generic_and is
begin

	and_computation	: process(and_in)

		variable and_var	: std_logic;
	begin
		and_var	:= '1';

		-- loop over all the input bits
		and_loop	: for in_bit in 0 to N-1
		loop

			and_var	:= and_var and and_in(in_bit);

		end loop and_loop;

		and_out	<= and_var;

	end process and_computation;

end architecture behaviour;
