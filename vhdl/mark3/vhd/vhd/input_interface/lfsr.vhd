library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity lfsr is

	generic(
		bit_width	: integer	:= 16
	);

	port(
		-- control input
		clk		: in std_logic;
		load		: in std_logic;
		update		: in std_logic;

		-- data input
		seed		: in unsigned(bit_width-1 downto 0);

		-- output
		pseudo_random	: out unsigned(bit_width-1 downto 0)
	);

end entity lfsr;

architecture behaviour_16bit of lfsr is

	signal reg_out		: unsigned(bit_width-1 downto 0);
	signal feedback		: std_logic;
	signal reg_en		: std_logic;

	
	component shift_register_unsigned is

		generic(
			bit_width	: integer := 16
		);

		port(
			-- input
			clk		: in std_logic;
			shift_in	: in std_logic;
			reg_en		: in std_logic;
			shift_en	: in std_logic;
			reg_in		: in unsigned(bit_width-1 downto 0);

			-- output
			reg_out		: out unsigned(bit_width-1 downto 0)
		);

	end component shift_register_unsigned;

begin

	pseudo_random	<= reg_out;

	feedback	<= reg_out(0) xor reg_out(1) xor reg_out(3) xor
		    		reg_out(12);

	reg_en		<= load or update;


	core_shift_register	: shift_register_unsigned 

		generic map(
			bit_width	=> bit_width
		)

		port map(
			-- input
			clk		=> clk,
			shift_in	=> feedback,
			reg_en		=> reg_en,
			shift_en	=> update,
			reg_in		=> seed,

			-- output
			reg_out		=> reg_out
		);

end architecture behaviour_16bit;
