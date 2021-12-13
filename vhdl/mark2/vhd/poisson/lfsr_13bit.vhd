library ieee;
use ieee.std_logic_1164.all;


entity lfsr_13bit is

	port(
		-- input
		clk		: in std_logic;
		en		: in std_logic;
		load_n		: in std_logic;
		lfsr_in		: in std_logic_vector(12 downto 0);

		-- output
		lfsr_out	: out std_logic_vector(12 downto 0)
	);

end entity lfsr_13bit;


architecture behaviour of lfsr_13bit is

	signal reg_out	: std_logic_vector(12 downto 0);
	signal feedback	: std_logic;

	component shift_register is

		generic(
			N		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			shift_in	: in std_logic;
			reg_en		: in std_logic;
			shift_en	: in std_logic;
			reg_in		: in std_logic_vector(N-1 downto 0);

			-- output
			reg_out		: out std_logic_vector(N-1 downto 0)
		);

	end component shift_register;



begin

	lfsr_out	<= reg_out;

	feedback	<= reg_out(0) xor reg_out(9) xor reg_out(10) xor
				reg_out(12);

	inner_shift_reg	: shift_register 

		generic map(
			N		=> 13
		)

		port map(
			-- input
			clk		=> clk,
			shift_in	=> feedback,
			reg_en		=> en,
			shift_en	=> load_n,
			reg_in		=> lfsr_in,

			-- output
			reg_out		=> reg_out
		);


end architecture behaviour;
