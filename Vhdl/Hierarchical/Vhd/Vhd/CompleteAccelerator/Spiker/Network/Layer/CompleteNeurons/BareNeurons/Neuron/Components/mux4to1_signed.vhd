library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mux4to1_signed is

	generic(
		-- bit-width
		N	: integer		
	);

	port(	
		-- inputs	
		sel	: in std_logic_vector(1 downto 0);
		in0	: in signed(N-1 downto 0);
		in1	: in signed(N-1 downto 0);
		in2	: in signed(N-1 downto 0);
		in3	: in signed(N-1 downto 0);

		-- output
		mux_out	: out signed(N-1 downto 0)
	);

end entity mux4to1_signed;



architecture behaviour of mux4to1_signed is
begin

	selection	: process(sel, in0, in1, in2, in3)
	begin
	
		case sel is
		
			when "00" =>
				mux_out <= in0;

			when "01" =>
				mux_out <= in1;

			when "10" =>
				mux_out <= in2;

			when others =>
				mux_out <= in3;

		end case;

	end process selection;

end architecture behaviour;
