library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity shifter is

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

end entity shifter;


architecture behaviour of shifter is
begin

	shifted_out(N-1 downto N-1-shift)	<= (others => shifter_in(N-1));

	shifted_out(N-1-shift downto 0)		<= shifter_in(N-1 downto shift);	

end architecture behaviour;
