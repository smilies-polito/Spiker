library ieee;
use ieee.std_logic_1164.all;

entity ff_tb is
end entity ff_tb;

architecture behaviour of ff_tb is

	-- input
	signal clk	: std_logic;
	signal ff_en	: std_logic;
	signal ff_in	: std_logic;

	-- output
	signal ff_out	: std_logic;

	component ff is

		port(
			-- input
			clk	: in std_logic;
			ff_en	: in std_logic;		
			ff_in	: in std_logic;

			-- output
			ff_out	: out std_logic
		);

	end component ff;

begin



	-- clock
	clock_gen 	: process
	begin
		clk	<= '0';				-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         		-- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- enable
	ff_en_gen	: process
	begin
		ff_en	<= '0';
		wait for 62 ns;
		ff_en <= '1';
		wait;
	end process ff_en_gen;


	-- input data
	ff_in_gen	: process
	begin
		ff_in	<= '0';				-- falling edge i*12ns
		wait for 12 ns;			                    
		ff_in	<= '1';         		-- rising edge 6ns + i*12ns
		wait for 12 ns;	
	end process ff_in_gen;




	dut	: ff
		port map(
			-- input
			clk	=> clk,
			ff_en	=> ff_en,
			ff_in	=> ff_in,

			-- output
			ff_out	=> ff_out
		);


end architecture behaviour;
