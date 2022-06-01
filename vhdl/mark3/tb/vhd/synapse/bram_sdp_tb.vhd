library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bram_sdp_tb is
end entity bram_sdp_tb;

architecture test of bram_sdp_tb is


	-- input
	signal clk		: std_logic;
	signal di		: std_logic_vector(71 downto 0);
	signal rdaddr		: std_logic_vector(8 downto 0);
	signal rden		: std_logic;
	signal wraddr		: std_logic_vector(8 downto 0);
	signal wren		: std_logic;

	-- output
	signal do		: std_logic_vector(71 downto 0);

	component bram_sdp is

		port(
			-- input
			clk		: in std_logic;
			di		: in std_logic_vector(71 downto 0);
			rdaddr		: in std_logic_vector(8 downto 0);
			rden		: in std_logic;
			wraddr		: in std_logic_vector(8 downto 0);
			wren		: in std_logic;

			-- output
			do		: out std_logic_vector(71 downto 0)
					
		);

	end component bram_sdp;

begin

	
	-- clock
	clock_gen : process
	begin
		clk	<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- wren
	wren_gen	: process
	begin
		wren	<= '0';
		wait for 14 ns;
		wren	<= '1';
		wait for 12 ns;
		wren	<= '0';
		wait;
	end process wren_gen;


	-- wraddr
	wraddr_gen	: process
	begin
		wraddr	<= (others => '0');
		wait for 14 ns;
		wraddr	<= std_logic_vector(to_unsigned(1, 9));
		wait for 12 ns;
		wraddr	<= (others => '0');
		wait;
	end process wraddr_gen;


	-- di
	di_gen	: process
	begin
		di	<= (others => '0');
		wait for 14 ns;
		di	<= (others => '1');
		wait for 12 ns;
		di	<= (others => '0');
	end process di_gen;


	-- rden
	rden_gen	: process
	begin
		rden	<= '0';
		wait for 26 ns;
		rden	<= '1';
		wait;
	end process rden_gen;


	-- rdaddr
	rdaddr_gen	: process
	begin
		rdaddr	<= (others => '0');
		wait for 26 ns;
		rdaddr	<= std_logic_vector(to_unsigned(1, 9));
		wait;
	end process rdaddr_gen;

	dut	: bram_sdp 

		port map(
			-- input
			clk	=> clk,
			di	=> di,
			rdaddr	=> rdaddr,
			rden	=> rden,
			wraddr	=> wraddr,
			wren	=> wren,

			-- output
			do	=> do
		);


end architecture test;
