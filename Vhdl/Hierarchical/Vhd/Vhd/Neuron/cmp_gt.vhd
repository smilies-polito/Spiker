library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cmp_gt is

	generic(
		-- bit-width
		N	: integer := 8		
	);

	port(
		-- input
		in0	: in signed(N-1 downto 0);
		in1	: in signed(N-1 downto 0);

		-- output
		cmp_out	: out std_logic
	);

end entity cmp_gt;


architecture behaviour of cmp_gt is
begin

	compare	: process(in0, in1)
	begin
		
		if in0 > in1
		then
			cmp_out <= '1';
		else
			cmp_out <= '0';
		end if;

	end process compare;

end architecture behaviour;
