library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity add_sub_tb is
end entity add_sub_tb;

architecture test of add_sub_tb is

	constant N		: integer := 8;			


	signal in0		: signed(N-1 downto 0);
	signal in1		: signed(N-1 downto 0);
	signal add_or_sub	: std_logic;
	signal add_sub_out	: signed(N-1 downto 0);
	
	
	component add_sub is

		generic(
			N		: integer := 8		
		);

		port(
			-- input
			in0		: in signed(N-1 downto 0);
			in1		: in signed(N-1 downto 0);
			add_or_sub	: in std_logic;

			-- output
			add_sub_out	: out signed(N-1 downto 0)		
		);

	end component add_sub;

begin




	simulate	: process
	begin

		in0		<= "00000001";
		in1		<= "00000001";
		add_or_sub	<= '0';

		wait for 10 ns;

		in0		<= "00000001";
		in1		<= "00000001";
		add_or_sub	<= '1';

		wait for 10 ns;

		in0		<= "11111111";
		in1		<= "11111111";
		add_or_sub	<= '0';

		wait for 10 ns;

		in0		<= "11111111";
		in1		<= "11111111";
		add_or_sub	<= '1';

		wait for 10 ns;

		in0		<= "01111111";
		in1		<= "01111111";
		add_or_sub	<= '0';

		wait for 10 ns;

		in0		<= "10000000";
		in1		<= "10000000";
		add_or_sub	<= '0';


		wait;


	end process simulate;


	dut	: add_sub
		generic map(
			N	=> 8		
		)

		port map(
			-- input
			in0		=> in0,
			in1		=> in1,
			add_or_sub	=> add_or_sub,
			
			-- output
			add_sub_out	=> add_sub_out		
		);


end architecture test;
