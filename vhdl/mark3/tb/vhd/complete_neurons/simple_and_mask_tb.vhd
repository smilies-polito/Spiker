library ieee;
use ieee.std_logic_1164.all;

entity simple_and_mask_tb is
end entity simple_and_mask_tb;


architecture behaviour of simple_and_mask_tb is

	constant N		: integer := 8;

	-- input
	signal input_bits	: std_logic_vector(N-1 downto 0);
	signal mask_bit		: std_logic;

	-- output
	signal output_bits	: std_logic_vector(N-1 downto 0);

	component simple_and_mask is

		generic(
			-- number of input bits on which the mask is applied
			N		: integer := 8		
		);

		port(
			-- input
			input_bits	: in std_logic_vector(N-1 downto 0);
			mask_bit	: in std_logic;

			-- output
			output_bits	: out std_logic_vector(N-1 downto 0)
		);

	end component simple_and_mask;


begin

	input_bits <= (others => '1');

	mask_bit_gen	: process
	begin

		mask_bit	<= '0';
		wait for 10 ns;
		mask_bit	<= '1';
		wait;

	end process mask_bit_gen;



	dut	: simple_and_mask 

		generic map(
			-- number of input bits on which the mask is applied
			N		=> N
		)

		port map(
			-- input
			input_bits	=> input_bits,
			mask_bit	=> mask_bit,

			-- output
			output_bits	=> output_bits
		);

end architecture behaviour;
