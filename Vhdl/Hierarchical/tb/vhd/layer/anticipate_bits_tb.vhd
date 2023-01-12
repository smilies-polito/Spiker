library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity anticipate_bits_tb is
end entity anticipate_bits_tb;


architecture test of anticipate_bits_tb is


	-- parallelism
	constant N		: integer := 8;

	-- input
	signal clk		: std_logic;
	signal bits_en		: std_logic;
	signal anticipate	: std_logic;
	signal input_bits	: std_logic_vector(N-1 downto 0);

	-- output
	signal output_bits	: std_logic_vector(N-1 downto 0);



	component anticipate_bits is

		generic(
			-- parallelism
			N		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			bits_en		: in std_logic;
			anticipate	: in std_logic;
			input_bits	: in std_logic_vector(N-1 downto 0);

			-- output
			output_bits	: out std_logic_vector(N-1 downto 0)	
		);

	end component anticipate_bits;


begin


	-- clock
	clock_gen : process
	begin
		clk	<= '0';			-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         	-- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;

	-- enable
	bits_en_gen : process
	begin
		bits_en	<= '0';		-- 0 ns
		wait for 26 ns;
		bits_en	<= '1';		-- 26 ns
		wait for 204 ns;
		bits_en	<= '0';		-- 230 ns
		wait;
	end process bits_en_gen;


	-- anticipate
	anticipate_gen : process
	begin
		anticipate	<= '0';		-- 0 ns
		wait for 26 ns;
		anticipate	<= '1';		-- 26 ns
		wait for 48 ns;
		anticipate	<= '0';		-- 74 ns
		wait for 48 ns;
		anticipate	<= '1';		-- 50 ns
		wait for 12 ns;
		anticipate	<= '0';		-- 62 ns
		wait;
	end process anticipate_gen;


	-- input_bits
	input_bits_gen : process
	begin

		for i in 0 to 2**N-1
		loop
			input_bits	<= std_logic_vector(to_unsigned(i, N));		-- 0 ns
			wait for 12 ns;
		end loop;

		wait;
	end process input_bits_gen;





	dut	: anticipate_bits

		generic map(
			-- parallelism
			N		=> N
		)

		port map(
			-- input
			clk		=> clk,
			bits_en		=> bits_en,
			anticipate	=> anticipate,
			input_bits	=> input_bits,

			-- output
			output_bits	=> output_bits
		);


end architecture test;
