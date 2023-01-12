library ieee;
use ieee.std_logic_1164.all;

entity simple_and_mask is

	generic(
		-- number of input bits on which the mask is applied
		N		: integer := 8		
	);

	port(
		-- input
		input_bits	: in std_logic_vector(N-1 downto 0);
		mask_bit	: in std_logic;

		-- output
		output_bits	: out std_logic_vector(N-1 downto 0)
	);

end entity simple_and_mask;


architecture behaviour of simple_and_mask is
begin

	mask	: process(input_bits, mask_bit)
	begin

		for i in 0 to N-1
		loop

			output_bits(i)	<= input_bits(i) and mask_bit;

		end loop;

	end process mask;

end architecture behaviour;
