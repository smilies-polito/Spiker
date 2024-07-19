library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity rom_128x10_exc_ip is
	port(
		clka	: in std_logic;
		addra	: in std_logic_vector(6 downto 0);
		douta	: out std_logic_vector(39 downto 0)
	);
end entity rom_128x10_exc_ip; 

architecture behavior of rom_128x10_exc_ip is

	type rom_type is array(0 to 128) of std_logic_vector(39 downto 0);

	constant mem	: rom_type := (
		"0000110100010110011010001011101011011110",
		"0000101000110000001000001110011011011111",
		"1111011011110000100111001011001000000010",
		"0000000100000000010011100000100100000011",
		"1011000100100000000011110010000100101110",
		"1100000111000001000000011110001000100001",
		"0001000000011000100000010011000100100000",
		"0000000000110001111011100000001010100011",
		"1101000111100001001100011101000011110000",
		"0000001000001100000000100010000011011101",
		"0000000111100001111100000010111000101110",
		"0000000100001110000000100001111011100010",
		"0000111111100000111100000010111111100001",
		"0000000100100010110100001110000111100000",
		"1110110100001010001011110010010011000000",
		"1100110101000010001010110011101100010001",
		"1110001000101011001011110000000000000010",
		"0010000100011110111000000001111100000000",
		"0001000111011111001000001100110000001111",
		"0000010011000001001000101101111010011101",
		"1110101000011010001000000010001100111010",
		"1101000100000010000000111100111000111110",
		"0100111111010000000011111110000111011110",
		"1001100100101100001000000001001100001010",
		"0000000001000000111010010010001011011100",
		"1111001011010000000111100000000011101111",
		"0000111100010100001100011011111000101011",
		"1101000011110000000111100001000000110010",
		"1111001100000010000100010000110011000000",
		"1111111000010001000000100000111011100001",
		"1100110101000000000011000011010011101100",
		"1001110100100011110011110001000100111111",
		"0010000000000000001000011110111111011011",
		"1010001000100000111111011100001000100001",
		"1101000011010001001011011111000100010000",
		"1111000000001111110100100001001011100000",
		"0000000100001110001000010001000011101110",
		"1001110000000011000011100000001000011111",
		"0010000011100001110011110000000000100000",
		"0011000111000000000111001101111100010000",
		"1111111000101111001000111101111101001110",
		"1110001000000001110100101111111111110001",
		"0010001011100000000000000000111011101110",
		"0000000000101111101001101000000000000010",
		"0000000110110000001000001111001000001011",
		"0011111100001100000011100001000000100000",
		"1100111100000000000011100001000000101100",
		"1111111011001111000011000010000011000100",
		"1010100001000000000000001100010001000010",
		"0100111011101001000010100110111011011111",
		"0010110010110001001011010010111111010001",
		"0010101100101110111100001101000100100101",
		"0000101000000000000000010101110100001101",
		"0000001000100000000010111111001011010010",
		"0011101111010100111111010111000011000011",
		"0000111000100000000110101111001000000011",
		"1111101100000000000100101101010100001110",
		"0010110000101101111111110000110000100000",
		"0001000000011101000010000000001111010001",
		"0000000011101101000000100000111000100000",
		"1100111011000000101101010010000101000000",
		"0010110100000001111100011110000000101110",
		"0000101011100011110000110000000111110010",
		"0100111111101111110000001011000000010000",
		"0001000000000010000011001111000011110001",
		"0000101111110001101100001110001101000011",
		"0001111000001100111011100000000011110001",
		"0000000000011110001000010001111011011111",
		"0011101011100001111100101101000000110000",
		"0000000000101110111000000000001111111111",
		"1011110011110001000111010001001100100011",
		"0000000111010001000000010000000111100000",
		"1110110000011110100101010011111100111110",
		"0001101100011011001110100011000001001111",
		"0011000000000010000011100000101111000001",
		"0000000000011110001000000010110100000000",
		"1101110000001111111100000100001100111101",
		"0000000111110000100000001101010000000100",
		"0010111100011101111011010001000011010001",
		"0001000111100001110000000000001000111110",
		"0001101110111100010000010000001111100000",
		"0000000110101111001011000000111100001110",
		"1111000011110010110100110000000000011111",
		"1111000111111101000101001111000000000000",
		"1110000011101111000000011111001011010010",
		"1101001111110001110111101110001000100010",
		"0000001000001110000011100010000011111011",
		"1111101100100000001100010001111001100011",
		"0000000000100011000011011111000111010010",
		"1110001011101111001011100000000100000001",
		"0010000100011110000000000010111100101110",
		"0000000000000000000000000000000000000000",
		"1010000000010000011100101101110000000000",
		"0000000011110000000111100011110111110010",
		"0000101000111000001000100010001100001100",
		"0001001000011110111111111111000000000000",
		"0010101010111110010110100110101000001100",
		"0010101100000010101100011100000100000011",
		"1110001000011111110100000000111000011111",
		"0001000111010001000111110001000100011101",
		"0010100000100100101000101111110100000011",
		"1100101100110010110100000100000000100001",
		"0010000100010000111111110000110111100000",
		"0011100100100000010000001110110000110010",
		"0000111100010001110100011110110001000000",
		"1000000001111101110100101101000100011100",
		"0001000000000001001000011110111111100000",
		"0001000111000000000100001101001111110001",
		"0001111111100000000000110001000100001100",
		"0001111100101101000100000001111111010011",
		"1100000011101110000111110000001000100000",
		"0001111100001110000000000010000011100000",
		"0000001011100011111111110001000111100000",
		"0001111101001111000000001110101000101011",
		"0001000000000010001011100010100111100001",
		"1101111100010001001011010001000000000001",
		"0010101100000000010000111010101111100010",
		"0000000000011110111000101111000011101110",
		"1111000011010010000101011011101101000000",
		"0000001011010001000010110000001000000001",
		"0011110000000000000000101101111111110001",
		"1010110100110000111000000001001000101011",
		"0000001000011110111100001110111100001110",
		"0010000011101101111100010000000111010001",
		"1101111100000000001100100010110000111011",
		"1100111101000000111110111111010001011001",
		"0010111100010000000000000010111011010001",
		"1001001000010011000100001101000011110000",
		"0000000000000000000000000000000000000000"
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
