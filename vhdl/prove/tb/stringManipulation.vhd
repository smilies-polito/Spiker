library ieee;
library std;
use ieee.std_logic_1164.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity string_man is
end entity string_man;

architecture behaviour of string_man is
begin

	string_manipulation	: process

		variable name		: string := "weights";
		variable ext		: string := ".mem";
		variable filename	: string;
		variable out_string	: line;
		variable i		: integer := 0;

	begin

		filename := name & i & ext;

		write(out_string, filename);
		writeline(output, filename);
		wait;

	end process string_manipulation;

end architecture behaviour;
