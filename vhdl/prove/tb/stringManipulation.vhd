library ieee;
library std;
use ieee.std_logic_1164.all;
use ieee.std_logic_textio.all;
use std.textio.all;

entity string_man is
end entity string_man;

architecture behaviour of string_man is

	constant name		: string := "weights";
	constant ext		: string := ".mem";

	file weights		: text;

begin

	string_manipulation	: process

		variable filename	: string(1 to 12);
		variable i		: integer; 
		variable i_string	: string(1 to 1);

	begin

		for i in 0 to 50
		loop

			i_string := integer'image(i)(0);
			filename := name & i_string & ext;

			file_open(weights, filename, read_mode);

			wait for 10 ns;
		end loop;

		wait;

	end process string_manipulation;

end architecture behaviour;
