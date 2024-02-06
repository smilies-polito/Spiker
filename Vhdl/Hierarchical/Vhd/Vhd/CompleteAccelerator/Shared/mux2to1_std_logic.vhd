library ieee;
use ieee.std_logic_1164.all;

entity mux2to1_std_logic is

	port(	
		-- inputs	
		sel	: in std_logic;
		in0	: in std_logic;
		in1	: in std_logic;

		-- output
		mux_out	: out std_logic
	);

end entity mux2to1_std_logic;



architecture behaviour of mux2to1_std_logic is
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
