library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity lfsr is

	generic(
		bit_width	: integer	:= 16
	);

	port(
		-- control input
		clk		: in std_logic;
		load		: in std_logic;
		update		: in std_logic;

		-- data input
		seed		: in unsigned(bit_width-1 downto 0);

		-- output
		pseudo_random	: out unsigned(bit_width-1 downto 0)
	);

end entity lfsr;

architecture behaviour_16bit of lfsr is
begin

end architecture behaviour_16bit;
