library ieee;
use ieee.std_logic_1164.all;

entity cmp_eq is

	generic(
		N	: integer := 8		
	);

	port(
		-- input
		in0	: in std_logic_vector(N-1 downto 0);
		in1	: in std_logic_vector(N-1 downto 0);

		-- output
		cmp_out	: out std_logic
	);

end entity cmp_eq;


architecture behaviour of cmp_eq is
begin

	compare	: process(in0, in1)
	begin

		if in0 = in1
		then
			cmp_out <= '1';
		else
			cmp_out <= '0';
		end if;

	end process compare;

end architecture behaviour;
