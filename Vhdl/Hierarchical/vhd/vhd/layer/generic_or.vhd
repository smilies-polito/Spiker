library ieee;
use ieee.std_logic_1164.all;

entity generic_or is

	generic(
		N	: integer := 8		
	);

	port(
		-- input
		or_in	: in std_logic_vector(N-1 downto 0);

		-- output
		or_out	: out std_logic
	);

end entity generic_or;


architecture behaviour of generic_or is
begin

	or_computation	: process(or_in)

		variable or_var	: std_logic;
	begin
		or_var	:= '0';

		-- loop over all the input bits
		or_loop	: for in_bit in 0 to N-1
		loop

			or_var	:= or_var or or_in(in_bit);

		end loop or_loop;

		or_out	<= or_var;

	end process or_computation;

end architecture behaviour;
