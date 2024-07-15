library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity rom_10x10_inh_ip is
	port(
		clka	: in std_logic;
		addra	: in std_logic_vector(3 downto 0);
		douta	: out std_logic_vector(9 downto 0)
	);
end entity rom_10x10_inh_ip; 

architecture behavior of rom_10x10_inh_ip is

	type rom_type is array(0 to 9) of std_logic_vector(9 downto 0);

	constant mem	: rom_type := (
		"0000000000",
		"0000000000",
		"0000000000",
		"0000000000",
		"0000000000",
		"0000000000",
		"0000000000",
		"0000000000",
		"0000000000",
		"0000000000"
	);

begin

	rom_behavior	: process(clka)
	begin

		if clka'event and clka = '1'
		then

			douta <= mem(to_integer(unsigned(addra)));

		end if;

	end process rom_behavior;

end architecture behavior;
