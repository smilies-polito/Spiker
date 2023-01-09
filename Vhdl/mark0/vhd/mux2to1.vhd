library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mux2to1 is

	generic(
		-- parallelism
		N	: integer		
	);

	port(	
		-- inputs	
		sel	: in std_logic;
		in0	: in signed(N-1 downto 0);
		in1	: in signed(N-1 downto 0);

		-- output
		mux_out	: out signed(N-1 downto 0)
	);

end entity mux2to1;



architecture behaviour of mux2to1 is
begin

	selection	: process(sel, in0, in1)
	begin
	
		case sel is
			
			when '0' =>
				mux_out <= in0;
		
			when others =>
				mux_out <= in1;

		end case;

	end process selection;

end architecture behaviour;
