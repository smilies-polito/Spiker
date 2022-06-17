library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity generic_mux is

	generic(
		N_sel		: integer := 8;
		bit_width	: integer := 8
	);

	port(
		-- input
		mux_in	: in std_logic_vector(2**N_sel*bit_width-1 downto 0);
		mux_sel	: in std_logic_vector(N_sel-1 downto 0);

		-- output
		mux_out	: out std_logic_vector(bit_width-1 downto 0)
	);

end entity generic_mux;


architecture behaviour of generic_mux is
begin

	input_selection	: process(mux_in, mux_sel)
	begin

		mux_out	<= mux_in((to_integer(unsigned(mux_sel))+1)*bit_width-1
			   downto to_integer(unsigned(mux_sel))*bit_width);

	end process input_selection;

end architecture behaviour;
