library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity decoder is

	generic(
		N	: integer := 8		
	);

	port(
		-- input
		encoded_in	: in std_logic_vector(N-1 downto 0);

		-- output
		decoded_out	: out  std_logic_vector(2**N -1 downto 0)
	);

end entity decoder;


architecture behaviour of decoder is
begin


	decode: process(encoded_in)
	begin
		decoded_out <= (others => '0');
		decoded_out(to_integer(unsigned(encoded_in))) <= '1';
	end process decode;

end architecture;
