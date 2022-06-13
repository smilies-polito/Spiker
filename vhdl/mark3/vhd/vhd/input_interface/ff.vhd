library ieee;
use ieee.std_logic_1164.all;

entity ff is

	port(
		-- input
		clk	: in std_logic;
		ff_en	: in std_logic;		
		ff_in	: in std_logic;

		-- output
		ff_out	: out std_logic
	);

end entity ff;

architecture behaviour of ff is
begin

	sample	: process(clk, ff_en, ff_in)
	begin

		if clk'event and clk = '1'
		then

			if ff_en = '1'
			then

				ff_out <= ff_in;

			end if;

		end if;

	end process sample;

end architecture behaviour;
