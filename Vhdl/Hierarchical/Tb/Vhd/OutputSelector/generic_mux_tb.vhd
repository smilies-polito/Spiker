library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity generic_mux_tb is
end entity generic_mux_tb;

architecture test of generic_mux_tb is

	constant N_sel		: integer := 3;
	constant bit_width	: integer := 8;

	-- input
	signal mux_in	: std_logic_vector(2**N_sel*bit_width-1 downto 0);
	signal mux_sel	: std_logic_vector(N_sel-1 downto 0);

	-- output
	signal mux_out	: std_logic_vector(bit_width-1 downto 0);

	component generic_mux is

		generic(
			N_sel		: integer := 8;
			bit_width	: integer := 8
		);

		port(
			-- input
			mux_in	: in std_logic_vector(2**N_sel*bit_width-1 downto 0);
			mux_sel	: in std_logic_vector(N_sel-1 downto 0);

			-- output
			mux_out	: out std_logic_vector(bit_width-1 downto 0)
		);

	end component generic_mux;

begin

	mux_in_gen	: process
	begin

		for i in 0 to 2**N_sel-1
		loop

			mux_in((i+1)*bit_width-1 downto i*bit_width) <=
				std_logic_vector(to_unsigned(i, bit_width));

		end loop;

		wait for 2**N_sel * 10 ns + 10 ns + 30 ns;

		mux_in(2**N_sel*bit_width - 1 downto (2**N_sel-1)*bit_width) <=
	       		std_logic_vector(to_unsigned(10, bit_width));

		wait for 10 ns;

		mux_in(2**N_sel*bit_width - 1 downto (2**N_sel-1)*bit_width) <=
	       		std_logic_vector(to_unsigned(11, bit_width));

		wait;

	end process mux_in_gen;

	mux_sel_gen	: process
	begin

		wait for 10 ns;

		for i in 0 to 2**N_sel-1
		loop

			mux_sel <= std_logic_vector(to_unsigned(i, N_sel));
			wait for 10 ns;

		end loop;

		wait;

	end process mux_sel_gen;


	dut	: generic_mux 

		generic map(
			N_sel		=> N_sel,
			bit_width	=> bit_width
		)

		port map(
			-- input
			mux_in		=> mux_in,
			mux_sel		=> mux_sel,

			-- output
			mux_out		=> mux_out
		);

end architecture test;
