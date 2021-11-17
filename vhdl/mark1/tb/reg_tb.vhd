library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity reg_tb is
end entity reg_tb;

architecture test of reg_tb is

	component reg is
		
		generic(
			-- parallelism
			N	: integer		
		);

		port(	
			-- inputs	
			clk	: in std_logic;
			en	: in std_logic;
			reg_in	: in signed(N-1 downto 0);

			-- outputs
			reg_out	: out signed(N-1 downto 0)
		);

	end component reg;


	constant N	: integer := 8;

	signal clk	: std_logic;
	signal en	: std_logic;
	signal reg_in	: signed(N-1 downto 0);
	signal reg_out	: signed(N-1 downto 0);

begin


	clock		: process
	begin
		clk	<= '0';
		wait for 10 ns;
		clk	<= '1';
		wait for 10 ns;
	end process clock;


	simulate	: process
	begin

		en	<= '0';
		reg_in	<= "00001111";

		wait for 15 ns;

		reg_in	<= "10101010";
		
		wait for 10 ns;

		en	<= '1';
		reg_in	<= "00001111";

		wait for 10 ns;

		reg_in	<= "10101010";

		wait;

	end process simulate;


	dut	: reg
		generic map(
			N	=> 8		
		)
		port map(
			-- inputs 	
			clk	=> clk,	 
			en	=> en,
			reg_in	=> reg_in,
			  
			-- output
			reg_out	=> reg_out
		);	

end architecture test;
