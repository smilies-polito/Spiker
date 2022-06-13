library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity unsigned_cmp_gt is
	
	generic(
		bit_width	: integer := 8		
	);


	port(
		-- input
		in0	: in unsigned(bit_width-1 downto 0);
		in1	: in unsigned(bit_width-1 downto 0);

		-- output		
		cmp_out	: out std_logic
	);

end entity unsigned_cmp_gt;


architecture behaviour of unsigned_cmp_gt is
begin

	compare	: process(in0, in1)
	begin
		
		if in0 > in1
		then
			cmp_out <= '1';
		else
			cmp_out <= '0';
		end if;

	end process compare;


end architecture behaviour;
