library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity shift_register_unsigned_tb is
end entity shift_register_unsigned_tb;

architecture behaviour of shift_register_unsigned_tb is


	constant bit_width	: integer := 16;
	constant seed_int	: integer := 5;

	-- input
	signal clk		: std_logic;
	signal shift_in		: std_logic;
	signal reg_en		: std_logic;
	signal shift_en		: std_logic;
	signal reg_in		: unsigned(bit_width-1 downto 0);

	-- output
	signal reg_out		: unsigned(bit_width-1 downto 0);



	component shift_register_unsigned is

		generic(
			bit_width	: integer := 8		
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

	shift_in	<= '1';

	-- clock
	clock_gen 	: process
	begin
		clk		<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk		<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- enable
	reg_en_gen	: process
	begin
		reg_en		<= '0';		-- 0 ns
		wait for 62 ns;
		reg_en 		<= '1';		-- 62 ns
		wait;
	end process reg_en_gen;


	-- shift_en
	shift_en_gen	: process
	begin
		shift_en	<= '0';		-- 0 ns
		wait for 110 ns;
		shift_en 	<= '1';		-- 110 ns
		wait;
	end process shift_en_gen;


	-- reg_in
	reg_in_gen	: process
	begin
		for i in 0 to 2**bit_width-1
		loop
			reg_in	<= to_unsigned(i, bit_width);
			wait for 12 ns;
		end loop;
	end process reg_in_gen;



	dut	: shift_register_unsigned

			generic map(
				bit_width	=> bit_width
			)

			port map(
				-- input
				clk		=> clk,
				shift_in	=> shift_in,
				reg_en		=> reg_en,
				shift_en	=> shift_en,
				reg_in		=> reg_in,

				-- output
				reg_out		=> reg_out
			);


end architecture behaviour;
