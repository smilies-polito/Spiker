library ieee;
use ieee.std_logic_1164.all;

entity double_and_mask_n is

	generic(
		N		: integer := 3
	);

	port(
		-- input
		input_bits	: in std_logic_vector(N-1 downto 0);
		mask_bit0	: in std_logic;
		mask_bit1	: in std_logic;

		-- output
		output_bits	: out std_logic_vector(N-1 downto 0)
	);

end entity double_and_mask_n;


architecture behaviour of double_and_mask_n is
begin

	mask	: process(input_bits, mask_bit0, mask_bit1)
	begin

		for i in 0 to N-1
		loop

			output_bits(i)	<= mask_bit1 and not(input_bits(i) and
					mask_bit0);

		end loop;

	end process mask;

end architecture behaviour;
