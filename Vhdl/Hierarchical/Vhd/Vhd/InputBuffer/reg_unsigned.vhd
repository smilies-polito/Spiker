library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity reg_unsigned is

	generic(
		-- parallelism
		N	: integer	:= 16		
	);

	port(	
		-- inputs	
		clk	: in std_logic;
		en	: in std_logic;
		reg_in	: in unsigned(N-1 downto 0);

		-- outputs
		reg_out	: out unsigned(N-1 downto 0)
	);

end entity reg_unsigned;



architecture behaviour of reg_unsigned is
begin

	sample	: process(clk, en)
	begin
	
		if clk'event and clk = '1'
		then
			if en = '1'
			then
				-- sample the input
				reg_out <= reg_in;
			end if;
		end if;
	end process sample;

end architecture behaviour;
