library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity reg_signed_sync_rst is

	generic(
		-- parallelism
		N	: integer	:= 16		
	);

	port(	
		-- inputs	
		clk	: in std_logic;
		en	: in std_logic;
		rst_n	: in std_logic;
		reg_in	: in signed(N-1 downto 0);

		-- outputs
		reg_out	: out signed(N-1 downto 0)
	);

end entity reg_signed_sync_rst;



architecture behaviour of reg_signed_sync_rst is
begin

	sample	: process(clk, en)
	begin
	
		
		if clk'event and clk = '1'
		then
			if rst_n = '0'
			then
				-- syncronous reset
				reg_out <= (others => '0');

			elsif en = '1'
			then
				-- sample the input
				reg_out <= reg_in;

			end if;
		end if;
	end process sample;

end architecture behaviour;
