library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity generic_mux_1bit is

	generic(
		N_sel	: integer := 8		
	);

	port(
		-- input
		mux_in	: in std_logic_vector(2**N_sel-1 downto 0);
		mux_sel	: in std_logic_vector(N_sel-1 downto 0);

		-- output
		mux_out	: out std_logic
	);

end entity generic_mux_1bit;


architecture behaviour of generic_mux_1bit is
begin

	input_selection	: process(mux_in, mux_sel)
	begin

		mux_out	<= mux_in(to_integer(unsigned(mux_sel)));

	end process input_selection;

end architecture behaviour;
