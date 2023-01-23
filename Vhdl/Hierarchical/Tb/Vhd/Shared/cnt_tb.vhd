library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity cnt_tb is
end entity cnt_tb;

architecture test of cnt_tb is

	constant bit_width		: integer := 4;

	signal clk		: std_logic;
	signal cnt_en		: std_logic;
	signal cnt_rst_n	: std_logic;

	signal cnt_out		: std_logic_vector(bit_width-1 downto 0);


	component cnt is

		generic(
			bit_width		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			cnt_en		: in std_logic;
			cnt_rst_n	: in std_logic;

			-- output
			cnt_out		: out std_logic_vector(bit_width-1 downto 0)		
		);

	end component cnt;


begin


	-- clock
	clock_gen : process
	begin
		clk	<= '0';			-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         	-- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;



	-- reset
	cnt_rst_n_gen : process
	begin
		cnt_rst_n	<= '1';		-- 0 ns
		wait for 14 ns;
		cnt_rst_n	<= '0';		-- 14 ns
		wait for 12 ns;
		cnt_rst_n	<= '1';		-- 26 ns
		wait;
	end process cnt_rst_n_gen;
	

	-- enable
	cnt_en_gen : process
	begin
		cnt_en	<= '0';		-- 0 ns
		wait for 26 ns;
		cnt_en	<= '1';		-- 26 ns
		wait for 204 ns;
		cnt_en	<= '0';		-- 230 ns
		wait;
	end process cnt_en_gen;



	dut	: cnt
		generic map(
			bit_width		=> bit_width
		)                                          
                                                           
		port map(                  
			-- input           
			clk		=> clk,
			cnt_en		=> cnt_en,
			cnt_rst_n	=> cnt_rst_n,
                                                           
			-- output          
			cnt_out		=> cnt_out
		);


end architecture test;
