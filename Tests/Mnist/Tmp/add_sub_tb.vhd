library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity add_sub_tb is
end entity add_sub_tb;

architecture behavior of add_sub_tb is

	constant N	: integer := 4;

	component add_sub is
	    generic (
	        saturate : string := "False";
		N : integer := 6
	    );
	    port (
		in0 		: in signed(N-1 downto 0);
		in1 		: in signed(N-1 downto 0);
		add_or_sub 	: in std_logic;
		add_sub_out 	: out signed(N-1 downto 0)
	    );
	end component add_sub;

	signal in0 		: signed(N-1 downto 0);
	signal in1 		: signed(N-1 downto 0);
	signal add_or_sub 	: std_logic;
	signal add_sub_out 	: signed(N-1 downto 0);

begin

	in0_in1_gen	: process

		variable as	: std_logic_vector(0 downto 0);
	begin

		for i in 0 to 1
		loop
			as := std_logic_vector(to_unsigned(i, 1));
			add_or_sub <= as(0);

			for j in -2**(N-1) to 2**(N-1)-1
			loop
				in0 <= to_signed(j, in0'length);

				for k in -2**(N-1) to 2**(N-1)-1
				loop
					in1 <= to_signed(k, in1'length);
					wait for 10 ns;
				end loop;
			end loop;
		end loop;

		wait;

	end process in0_in1_gen;

	dut	: add_sub
	    generic map(
	    	saturate	=> "True",
		N 		=> N
	    )
	    port map(
		in0 		=> in0,
		in1 		=> in1, 	   
		add_or_sub 	=> add_or_sub,
		add_sub_out 	=> add_sub_out
	    );

end architecture behavior;
