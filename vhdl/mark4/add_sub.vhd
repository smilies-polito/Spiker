library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity add_sub is

	generic(
		N		: integer := 8		
	);

	port(
		-- input
		in0		: in signed(N-1 downto 0);
		in1		: in signed(N-1 downto 0);
		add_or_sub	: in std_logic;

		-- output
		add_sub_out	: out signed(N-1 downto 0)		
	);

end entity add_sub;


architecture behaviour of add_sub is
begin


	operation	: process(in0, in1, add_or_sub)
	begin

		if add_or_sub = '0'
		then
			add_sub_out	<= in0 + in1;
		else
			add_sub_out	<= in0 - in1;
		end if;

	end process operation;


end architecture behaviour;
