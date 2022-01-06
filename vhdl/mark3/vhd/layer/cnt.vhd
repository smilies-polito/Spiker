library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity cnt is

	generic(
		N		: integer := 8		
	);

	port(
		-- input
		clk		: in std_logic;
		cnt_en		: in std_logic;
		cnt_rst_n	: in std_logic;

		-- output
		cnt_out		: out std_logic_vector(N-1 downto 0)		
	);

end entity cnt;


architecture behaviour of cnt is
begin

	count	: process(clk)

		variable cnt_var	: integer	:= 0;
	begin

		if clk'event and clk = '1'
		then
			if cnt_rst_n = '0'
			then
				cnt_var	:= 0;
			else
				if cnt_en = '1'
				then
					cnt_var	:= cnt_var + 1;
				end if;
			end if;
		end if;

		cnt_out	<= std_logic_vector(to_unsigned(cnt_var, N));

	end process count;

	
end behaviour;
