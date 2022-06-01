library ieee;
use ieee.std_logic_1164.all;

entity double_and_mask_n_tb is
end entity double_and_mask_n_tb;


architecture behaviour of double_and_mask_n_tb is

	constant N		: integer := 8;

	-- input
	signal input_bits	: std_logic_vector(N-1 downto 0);
	signal mask_bit0	: std_logic;
	signal mask_bit1	: std_logic;

	-- output
	signal output_bits	: std_logic_vector(N-1 downto 0);

	component double_and_mask_n is

		generic(
			-- number of input bits on which the mask is applied
			N		: integer := 8		
		);

		port(
			-- input
			input_bits	: in std_logic_vector(N-1 downto 0);
			mask_bit0	: in std_logic;
			mask_bit1	: in std_logic;

			-- output
			output_bits	: out std_logic_vector(N-1 downto 0)
		);

	end component double_and_mask_n;


begin

	input_bits <= (others => '1');

	mask_bit0_gen	: process
	begin

		mask_bit0	<= '1';
		wait for 10 ns;
		mask_bit0	<= '0';
		wait for 20 ns;
		mask_bit0	<= '1';
		wait;

	end process mask_bit0_gen;


	mask_bit1_gen	: process
	begin

		mask_bit1	<= '0';
		wait for 20 ns;
		mask_bit1	<= '1';
		wait;

	end process mask_bit1_gen;


	dut	: double_and_mask_n 

		generic map(
			-- number of input bits on which the mask is applied
			N		=> N
		)

		port map(
			-- input
			input_bits	=> input_bits,
			mask_bit0	=> mask_bit0,
			mask_bit1	=> mask_bit1,

			-- output
			output_bits	=> output_bits
		);

end architecture behaviour;
