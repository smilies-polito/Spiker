library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity cmp_eq_tb is
end entity cmp_eq_tb;

architecture test of cmp_eq_tb is

	constant N	: integer := 4;

	signal in0	: std_logic_vector(N-1 downto 0);
	signal in1	: std_logic_vector(N-1 downto 0);
	signal cmp_out	: std_logic;

	
	component cmp_eq is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			in0	: in std_logic_vector(N-1 downto 0);
			in1	: in std_logic_vector(N-1 downto 0);

			-- output
			cmp_out	: out std_logic
		);

	end component cmp_eq;


begin


	input_gen	: process
	begin

		for i in 0 to 2**N-1
		loop

			in0	<= std_logic_vector(to_unsigned(i, N));
			in1	<= std_logic_vector(to_unsigned(2**N-1-i, N));
			wait for 10 ns;

		end loop;

		wait for 10 ns;
		in1	<= "1111";

		wait;

	end process input_gen;




	dut	: cmp_eq
		generic map(
			N	=> N		
		)

		port map(
			-- input
			in0	=> in0,
			in1	=> in1,

			-- output
			cmp_out	=> cmp_out
		);


end architecture test;
