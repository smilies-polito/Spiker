library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity reg_signed_sync_rst_tb is
end entity reg_signed_sync_rst_tb;

architecture test of reg_signed_sync_rst_tb is

	component reg_signed_sync_rst is
		
		generic(
			-- parallelism
			N	: integer		
		);

		port(	
			-- inputs	
			clk	: in std_logic;
			en	: in std_logic;
			rst_n	: in std_logic;
			reg_in	: in signed(N-1 downto 0);

			-- outputs
			reg_out	: out signed(N-1 downto 0)
		);

	end component reg_signed_sync_rst;


	constant N	: integer := 8;

	signal clk	: std_logic;
	signal en	: std_logic;
	signal rst_n	: std_logic;
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
		rst_n	<= '1';
		reg_in	<= "00001111";

		wait for 15 ns;

		reg_in	<= "10101010";
		
		wait for 10 ns;

		en	<= '1';
		reg_in	<= "00001111";

		wait for 10 ns;

		reg_in	<= "10101010";

		wait for 10 ns;

		rst_n <= '0';

		wait for 10 ns;

		en	<= '1';
		reg_in	<= "00001111";

		wait for 10 ns;
		rst_n <= '0';

		wait;

	end process simulate;


	dut	: reg_signed_sync_rst
		generic map(
			N	=> 8		
		)
		port map(
			-- inputs 	
			clk	=> clk,	 
			en	=> en,
			rst_n	=> rst_n,
			reg_in	=> reg_in,
			  
			-- output
			reg_out	=> reg_out
		);	

end architecture test;
