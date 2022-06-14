library ieee;
use ieee.std_logic_1164.all;


entity out_interface_cu_tb is
end entity out_interface_cu_tb;


architecture behaviour of out_interface_cu_tb is

	-- input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal start		: std_logic;
	signal stop		: std_logic;

	-- output
	signal elaborate	: std_logic;
	signal cnt_rst_n	: std_logic;


	component out_interface_cu is

		port(
			-- input
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			stop		: in std_logic;

			-- output
			elaborate	: out std_logic;
			cnt_rst_n	: out std_logic
		);

	end component out_interface_cu;

begin

	dut	: out_interface_cu 

		port map(
			-- input
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			stop		=> stop,

			-- output
			elaborate	=> elaborate,
			cnt_rst_n	=> cnt_rst_n
		);
	

end architecture behaviour;
